"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# 明确从项目根目录读取 .env，避免受当前终端目录影响
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


@dataclass(frozen=True)
class Settings:
    """Configuration required by the LLM client."""

    api_key: str
    model: str
    base_url: str | None = None


def get_settings() -> Settings:
    """
    Read and validate application settings.

    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    missing_variables: list[str] = []

    if not api_key:
        missing_variables.append("LLM_API_KEY")

    if not model:
        missing_variables.append("LLM_MODEL")

    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise ValueError(
            f"缺少必要的环境变量：{missing_text}。"
            f"请检查配置文件：{ENV_FILE}"
        )

    return Settings(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


if __name__ == "__main__":
    settings = get_settings()

    print("配置读取成功")
    print(f"模型名称：{settings.model}")
    print(f"API 地址：{settings.base_url or '使用 SDK 默认地址'}")
    print(f"API Key 是否存在：{bool(settings.api_key)}")