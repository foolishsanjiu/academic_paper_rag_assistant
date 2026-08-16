"""PDF document loading and basic quality checking."""

from pathlib import Path

import pymupdf


def detect_text_warnings(text: str) -> list[str]:
    """
    Perform conservative checks for suspicious extracted text.

    These warnings only indicate that a page should be inspected.
    They do not prove that the PDF is corrupted.
    """
    warnings: list[str] = []

    if not text:
        warnings.append("empty_text")
        return warnings

    # Unicode replacement character often indicates decoding problems.
    replacement_count = text.count("\ufffd")

    if replacement_count > 0:
        warnings.append(
            f"replacement_characters:{replacement_count}"
        )

    # NULL characters are unusual in normal extracted paper text.
    null_count = text.count("\x00")

    if null_count > 0:
        warnings.append(
            f"null_characters:{null_count}"
        )

    # Private-use Unicode characters may originate from custom PDF fonts.
    private_use_count = sum(
        1
        for character in text
        if "\ue000" <= character <= "\uf8ff"
    )

    if private_use_count > 0:
        warnings.append(
            f"private_use_characters:{private_use_count}"
        )

    return warnings


def load_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """
    Read one PDF page by page and preserve page metadata.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 文件不存在：{pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"路径不是文件：{pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"不是 PDF 文件：{pdf_path}"
        )

    pages: list[dict] = []

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):

            raw_text = page.get_text(
                "text",
                sort=False,
            )

            text = raw_text.strip()

            warnings = detect_text_warnings(text)

            # PDF page label may be:
            # "i", "ii", "1", "2", ...
            page_label = page.get_label() or None

            page_info = {
                "text": text,
                "metadata": {
                    "file_name": pdf_path.name,

                    # PyMuPDF / Python 内部索引
                    "page_index": page_index,

                    # 用户通常理解的第 1、2、3... 个 PDF 页面
                    "page_number": page_index + 1,

                    # PDF 自身定义的页面标签
                    "page_label": page_label,

                    "char_count": len(text),

                    "is_empty": not bool(text),

                    "parse_warnings": warnings,
                },
            }

            pages.append(page_info)

    return pages


def load_pdf_directory(
    paper_dir: Path,
) -> list[dict]:
    """Read all PDFs in a directory."""
    if not paper_dir.exists():
        raise FileNotFoundError(
            f"论文目录不存在：{paper_dir}"
        )

    if not paper_dir.is_dir():
        raise NotADirectoryError(
            f"路径不是目录：{paper_dir}"
        )

    all_pages: list[dict] = []

    pdf_paths = sorted(
        paper_dir.glob("*.pdf")
    )

    if not pdf_paths:
        raise FileNotFoundError(
            f"目录中没有 PDF 文件：{paper_dir}"
        )

    for pdf_path in pdf_paths:
        print(f"正在读取：{pdf_path.name}")

        pages = load_pdf_pages(pdf_path)

        all_pages.extend(pages)

    return all_pages


def print_quality_summary(
    pages: list[dict],
) -> None:
    """Print a simple extraction-quality report."""
    empty_pages = [
        page
        for page in pages
        if page["metadata"]["is_empty"]
    ]

    warning_pages = [
        page
        for page in pages
        if page["metadata"]["parse_warnings"]
    ]

    print()
    print("=" * 70)
    print("PDF 解析质量汇总")
    print("=" * 70)

    print(f"总页数：{len(pages)}")
    print(f"空文本页：{len(empty_pages)}")
    print(f"存在解析警告的页：{len(warning_pages)}")

    if empty_pages:
        print("\n空文本页：")

        for page in empty_pages:
            metadata = page["metadata"]

            print(
                f"- {metadata['file_name']} "
                f"| PDF page {metadata['page_number']}"
            )

    if warning_pages:
        print("\n解析警告：")

        for page in warning_pages:
            metadata = page["metadata"]

            print(
                f"- {metadata['file_name']} "
                f"| PDF page {metadata['page_number']} "
                f"| {metadata['parse_warnings']}"
            )


def main() -> None:
    project_root = Path(__file__).resolve().parent

    paper_dir = (
        project_root
        / "data"
        / "papers"
    )

    pages = load_pdf_directory(
        paper_dir
    )

    print_quality_summary(pages)

    print()
    print("=" * 70)
    print("页面样例")
    print("=" * 70)

    for page in pages[:3]:
        metadata = page["metadata"]

        print()
        print(
            f"文件：{metadata['file_name']}"
        )

        print(
            f"PDF物理页码：{metadata['page_number']}"
        )

        print(
            f"PDF页面标签：{metadata['page_label']}"
        )

        print(
            f"字符数：{metadata['char_count']}"
        )

        print(
            f"解析警告：{metadata['parse_warnings']}"
        )

        print("-" * 70)

        print(
            page["text"][:800]
        )


if __name__ == "__main__":
    main()