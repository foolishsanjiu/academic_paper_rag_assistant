"""An OpenAI-compatible LLM client with streaming and logging."""

from collections.abc import Iterator
import logging
from time import perf_counter

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config import Settings, get_settings
from logging_config import setup_logging


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一名严谨的学术论文助手。"
    "回答应当准确、清晰，不编造不存在的信息。"
)


class LLMClient:
    """Client used to call an OpenAI-compatible language-model API."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the API client.

        Args:
            settings: Validated API configuration.
        """
        self.settings = settings

        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=30.0,
            max_retries=2,
        )

    @staticmethod
    def _validate_message(user_message: str) -> str:
        """
        Remove surrounding whitespace and reject empty input.
        """
        cleaned_message = user_message.strip()

        if not cleaned_message:
            raise ValueError("输入内容不能为空。")

        return cleaned_message

    @staticmethod
    def _build_messages(user_message: str) -> list[dict[str, str]]:
        """
        Build the message list sent to the model.
        """
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

    @staticmethod
    def _convert_api_error(error: Exception) -> RuntimeError:
        """
        Convert SDK exceptions into user-friendly application errors.
        """
        if isinstance(error, AuthenticationError):
            return RuntimeError(
                "API 身份验证失败，请检查 LLM_API_KEY。"
            )

        if isinstance(error, RateLimitError):
            return RuntimeError(
                "请求频率过高、余额不足或账户受到限流。"
            )

        if isinstance(error, APITimeoutError):
            return RuntimeError(
                "API 请求超时，请稍后重试。"
            )

        if isinstance(error, APIConnectionError):
            return RuntimeError(
                "无法连接 API，请检查网络和 LLM_BASE_URL。"
            )

        if isinstance(error, APIStatusError):
            return RuntimeError(
                f"API 返回异常状态码：{error.status_code}。"
            )

        return RuntimeError(
            f"调用模型时发生未知错误：{type(error).__name__}。"
        )

    def chat(
        self,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> str:
        """
        Send a non-streaming request and return the complete answer.

        Args:
            user_message: User input.

        Returns:
            Complete model answer.
        """
        cleaned_message = self._validate_message(user_message)
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0。")
        start_time = perf_counter()

        logger.info(
            "开始非流式请求 | model=%s | input_length=%d",
            self.settings.model,
            len(cleaned_message),
        )

        try:
            request: dict = {
                "model": self.settings.model,
                "messages": self._build_messages(cleaned_message),
                "stream": False,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request["max_tokens"] = max_tokens
            if thinking is not None:
                request["extra_body"] = {
                    "thinking": {
                        "type": "enabled" if thinking else "disabled"
                    }
                }

            response = self.client.chat.completions.create(**request)

            answer = response.choices[0].message.content

            if not answer:
                raise RuntimeError("模型返回了空内容。")

            elapsed_time = perf_counter() - start_time

            logger.info(
                "非流式请求成功 | model=%s | elapsed=%.2fs",
                self.settings.model,
                elapsed_time,
            )

            return answer.strip()

        except (
            AuthenticationError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ) as error:
            elapsed_time = perf_counter() - start_time

            logger.error(
                "非流式请求失败 | error=%s | elapsed=%.2fs",
                type(error).__name__,
                elapsed_time,
            )

            raise self._convert_api_error(error) from error

    def chat_json(self, user_message: str) -> str:
        """Request one non-streaming JSON object from the model."""
        cleaned_message = self._validate_message(user_message)
        start_time = perf_counter()

        logger.info(
            "开始结构化 JSON 请求 | model=%s | input_length=%d",
            self.settings.model,
            len(cleaned_message),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=self._build_messages(cleaned_message),
                response_format={"type": "json_object"},
                stream=False,
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            if not answer:
                raise RuntimeError("模型返回了空 JSON 内容。")

            logger.info(
                "结构化 JSON 请求成功 | model=%s | elapsed=%.2fs",
                self.settings.model,
                perf_counter() - start_time,
            )
            return answer.strip()

        except (
            AuthenticationError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ) as error:
            logger.error(
                "结构化 JSON 请求失败 | error=%s | elapsed=%.2fs",
                type(error).__name__,
                perf_counter() - start_time,
            )
            raise self._convert_api_error(error) from error

    def stream_chat(
        self,
        user_message: str,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """
        Send a streaming request and yield answer fragments.

        Args:
            user_message: User input.

        Yields:
            Text fragments returned by the model.
        """
        cleaned_message = self._validate_message(user_message)
        start_time = perf_counter()

        logger.info(
            "开始流式请求 | model=%s | input_length=%d",
            self.settings.model,
            len(cleaned_message),
        )

        received_content = False

        try:
            stream = self.client.chat.completions.create(
                model=self.settings.model,
                messages=self._build_messages(cleaned_message),
                stream=True,
                temperature=temperature,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    received_content = True
                    yield content

            if not received_content:
                raise RuntimeError("模型没有返回可用的流式内容。")

            elapsed_time = perf_counter() - start_time

            logger.info(
                "流式请求成功 | model=%s | elapsed=%.2fs",
                self.settings.model,
                elapsed_time,
            )

        except (
            AuthenticationError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
        ) as error:
            elapsed_time = perf_counter() - start_time

            logger.error(
                "流式请求失败 | error=%s | elapsed=%.2fs",
                type(error).__name__,
                elapsed_time,
            )

            raise self._convert_api_error(error) from error


def main() -> None:
    """Run an interactive streaming or non-streaming test."""
    setup_logging()

    try:
        settings = get_settings()
        llm = LLMClient(settings)

        print("请选择调用方式：")
        print("1. 非流式输出")
        print("2. 流式输出")

        mode = input("请输入 1 或 2：").strip()
        question = input("请输入问题：")

        if mode == "1":
            print("\n正在调用模型……")

            answer = llm.chat(question)

            print("\n模型回答：")
            print(answer)

        elif mode == "2":
            print("\n模型回答：")

            for content in llm.stream_chat(question):
                print(content, end="", flush=True)

            print()

        else:
            raise ValueError("调用方式只能选择 1 或 2。")

    except (ValueError, RuntimeError) as error:
        logger.error("程序执行失败 | message=%s", error)
        print(f"\n程序运行失败：{error}")


if __name__ == "__main__":
    main()
