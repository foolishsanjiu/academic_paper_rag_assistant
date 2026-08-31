"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# 明确从项目根目录读取 .env，避免受当前终端目录影响
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

PAPER_DIRECTORY = PROJECT_ROOT / "data" / "papers"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
INDEX_MANIFEST_PATH = CHROMA_DIRECTORY / "index_manifest.json"
LOG_DIRECTORY = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIRECTORY / "app.log"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_COLLECTION_NAME = "academic_papers"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_BM25_EPSILON = 0.25

DEFAULT_RRF_K = 60
DEFAULT_HYBRID_CANDIDATE_K = 20
DEFAULT_HYBRID_FUSION_K = 40

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_RERANK_CANDIDATE_K = 40

MULTI_DOCUMENT_TOP_K = 8
MULTI_DOCUMENT_CANDIDATE_K = 32
MULTI_DOCUMENT_MAX_CHUNKS_PER_FILE = 3

MAX_UPLOAD_SIZE_MB = 50

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
