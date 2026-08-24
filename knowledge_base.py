"""Knowledge-base PDF validation and management utilities."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pymupdf

from config import PAPER_DIRECTORY
from document_loader import compute_document_id


DEFAULT_PAPER_DIRECTORY = PAPER_DIRECTORY


# ============================================================
# Knowledge-base metadata
# ============================================================


def list_pdf_files(
    paper_directory: Path = DEFAULT_PAPER_DIRECTORY,
) -> list[Path]:
    """
    Return all PDF files in the knowledge-base directory.
    """
    if not paper_directory.exists():
        return []

    return sorted(
        paper_directory.glob("*.pdf")
    )


def get_paper_count(
    paper_directory: Path = DEFAULT_PAPER_DIRECTORY,
) -> int:
    """
    Return the number of PDF papers in the knowledge base.
    """
    return len(
        list_pdf_files(
            paper_directory
        )
    )


def get_paper_names(
    paper_directory: Path = DEFAULT_PAPER_DIRECTORY,
) -> list[str]:
    """
    Return knowledge-base PDF filenames.
    """
    return [
        path.name
        for path in list_pdf_files(
            paper_directory
        )
    ]


# ============================================================
# PDF validation
# ============================================================


@dataclass(frozen=True)
class PDFValidationResult:
    """Result of PDF upload validation."""

    file_name: str
    document_id: str
    size_bytes: int
    page_count: int
    text_char_count: int


def compute_bytes_document_id(
    data: bytes,
) -> str:
    """
    Compute the same style of SHA-256 document ID
    directly from uploaded bytes.
    """
    return hashlib.sha256(
        data
    ).hexdigest()[:16]


def validate_pdf_bytes(
    file_name: str,
    data: bytes,
) -> PDFValidationResult:
    """
    Validate one uploaded PDF before saving it.

    Checks:
    - PDF extension
    - non-empty file
    - PDF magic header
    - PyMuPDF can open the file
    - file contains at least one page
    - file contains extractable text
    """
    safe_name = Path(
        file_name
    ).name

    if not safe_name.lower().endswith(
        ".pdf"
    ):
        raise ValueError(
            "仅支持 PDF 文件。"
        )

    if not data:
        raise ValueError(
            "上传文件为空。"
        )

    if not data.startswith(
        b"%PDF-"
    ):
        raise ValueError(
            "文件扩展名虽然是 .pdf，"
            "但文件内容不是有效 PDF。"
        )

    try:
        with pymupdf.open(
            stream=data,
            filetype="pdf",
        ) as document:

            page_count = (
                document.page_count
            )

            if page_count <= 0:
                raise ValueError(
                    "PDF 中没有有效页面。"
                )

            text_char_count = 0

            for page in document:
                text = page.get_text(
                    "text",
                    sort=False,
                )

                text_char_count += len(
                    text.strip()
                )

    except pymupdf.FileDataError as error:
        raise ValueError(
            "PDF 文件损坏或无法解析。"
        ) from error

    if text_char_count == 0:
        raise ValueError(
            "PDF 中没有可提取文本。"
            "当前版本暂不支持纯扫描型 PDF。"
        )

    return PDFValidationResult(
        file_name=safe_name,
        document_id=(
            compute_bytes_document_id(
                data
            )
        ),
        size_bytes=len(data),
        page_count=page_count,
        text_char_count=(
            text_char_count
        ),
    )


# ============================================================
# Duplicate detection
# ============================================================


def get_existing_document_ids(
    paper_directory: Path = (
        DEFAULT_PAPER_DIRECTORY
    ),
) -> dict[str, str]:
    """
    Return:
        document_id -> file_name
    """
    result: dict[str, str] = {}

    if not paper_directory.exists():
        return result

    for pdf_path in list_pdf_files(
        paper_directory
    ):
        document_id = (
            compute_document_id(
                pdf_path
            )
        )

        result[
            document_id
        ] = pdf_path.name

    return result


# ============================================================
# PDF saving
# ============================================================


def save_uploaded_pdf(
    file_name: str,
    data: bytes,
    paper_directory: Path = (
        DEFAULT_PAPER_DIRECTORY
    ),
) -> tuple[
    Path,
    PDFValidationResult,
]:
    """
    Validate and safely save one uploaded PDF.
    """
    validation = (
        validate_pdf_bytes(
            file_name=file_name,
            data=data,
        )
    )

    existing_ids = (
        get_existing_document_ids(
            paper_directory
        )
    )

    if (
        validation.document_id
        in existing_ids
    ):
        existing_name = (
            existing_ids[
                validation.document_id
            ]
        )

        raise ValueError(
            "该 PDF 已存在于知识库中："
            f"{existing_name}"
        )

    paper_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_path = (
        paper_directory
        / validation.file_name
    )

    if target_path.exists():
        raise FileExistsError(
            "知识库中已经存在同名文件："
            f"{validation.file_name}"
            "。请先修改文件名。"
        )

    target_path.write_bytes(
        data
    )

    return (
        target_path,
        validation,
    )
