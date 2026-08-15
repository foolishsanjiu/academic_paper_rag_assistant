"""PDF document loading utilities."""

from pathlib import Path

import pymupdf


def load_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """
    Read a PDF page by page.

    Args:
        pdf_path:
            Path to the PDF file.

    Returns:
        A list of dictionaries containing
        file name, page number and page text.
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

            text = page.get_text("text").strip()

            page_info = {
                "file_name": pdf_path.name,

                # PyMuPDF 内部从 0 开始，
                # 这里转换成人类习惯的 1 开始
                "page_number": page_index + 1,

                "text": text,
            }

            pages.append(page_info)

    return pages


def load_pdf_directory(
    paper_dir: Path,
) -> list[dict]:
    """
    Read all PDF files in a directory.
    """
    if not paper_dir.exists():
        raise FileNotFoundError(
            f"论文目录不存在：{paper_dir}"
        )

    all_pages: list[dict] = []

    for pdf_path in sorted(
        paper_dir.glob("*.pdf")
    ):
        print(
            f"正在读取：{pdf_path.name}"
        )

        pages = load_pdf_pages(pdf_path)

        all_pages.extend(pages)

    return all_pages


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

    print()
    print(
        f"共读取 {len(pages)} 页"
    )

    for page in pages:

        print()
        print("=" * 70)

        print(
            f"文件：{page['file_name']}"
        )

        print(
            f"页码：{page['page_number']}"
        )

        print(
            f"字符数：{len(page['text'])}"
        )

        print("-" * 70)

        # 为避免终端一次输出几百页全文，
        # 暂时显示每页前 600 个字符。
        preview = page["text"][:600]

        print(preview)

        if len(page["text"]) > 600:
            print("\n......")


if __name__ == "__main__":
    main()