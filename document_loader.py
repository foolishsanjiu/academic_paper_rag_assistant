"""PDF document loading and basic quality checking."""

from pathlib import Path

import pymupdf
from langchain_core.documents import Document


def detect_text_warnings(text: str) -> list[str]:
    """
    Perform conservative checks for suspicious extracted text.

    These warnings only indicate that a page should be inspected.
    They do not prove that the PDF is corrupted.

    Args:
        text:
            Extracted page text.

    Returns:
        A list of warning strings.
    """
    warnings: list[str] = []

    if not text:
        warnings.append("empty_text")
        return warnings

    # Unicode replacement character.
    # This may indicate decoding or font-mapping problems.
    replacement_count = text.count("\ufffd")

    if replacement_count > 0:
        warnings.append(
            f"replacement_characters:{replacement_count}"
        )

    # NULL characters are unusual in normal extracted text.
    null_count = text.count("\x00")

    if null_count > 0:
        warnings.append(
            f"null_characters:{null_count}"
        )

    # Unicode Private Use Area characters may come from
    # custom PDF fonts, equations, or special symbols.
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
) -> list[Document]:
    """
    Read one PDF page by page and convert each page
    into a LangChain Document.

    Each Document contains:

    - page_content:
        Extracted text from one PDF page.

    - metadata:
        File name, page information, text length,
        empty-page status and parsing warnings.

    Args:
        pdf_path:
            Path to one PDF file.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError:
            If the PDF file does not exist.

        ValueError:
            If the path is not a file or is not a PDF.
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

    documents: list[Document] = []

    with pymupdf.open(pdf_path) as pdf:

        for page_index, page in enumerate(pdf):

            # 根据第 9 天的实验，
            # 当前暂时使用 sort=False。
            #
            # 对测试的 IEEE 双栏论文来说，
            # sort=False 更适合形成连续的 RAG 文本。
            raw_text = page.get_text(
                "text",
                sort=False,
            )

            text = raw_text.strip()

            warnings = detect_text_warnings(text)

            # PDF 自身定义的逻辑页标签。
            #
            # 例如：
            # i, ii, iii, 1, 2, 3 ...
            #
            # 如果 PDF 没有定义页面标签，
            # get_label() 可能返回空字符串。
            page_label = page.get_label() or None

            document = Document(
                page_content=text,
                metadata={
                    # 原始论文文件名
                    "file_name": pdf_path.name,

                    # PyMuPDF / Python 内部页索引：
                    # 0, 1, 2, ...
                    "page_index": page_index,

                    # 用户看到的 PDF 物理页：
                    # 1, 2, 3, ...
                    "page_number": page_index + 1,

                    # PDF 内部定义的页面标签。
                    # 例如 "iii"、"1"、"25"
                    "page_label": page_label,

                    # 当前页面提取出的字符数
                    "char_count": len(text),

                    # 当前页是否没有提取到文本
                    "is_empty": not bool(text),

                    # 第 9 天加入的解析质量警告
                    "parse_warnings": warnings,
                },
            )

            documents.append(document)

    return documents


def load_pdf_directory(
    paper_dir: Path,
) -> list[Document]:
    """
    Read all PDF files in one directory.

    Each PDF page is converted into one LangChain Document.

    Args:
        paper_dir:
            Directory containing PDF papers.

    Returns:
        A list containing Documents from all PDF files.

    Raises:
        FileNotFoundError:
            If the directory does not exist or contains no PDF files.

        NotADirectoryError:
            If the given path is not a directory.
    """
    if not paper_dir.exists():
        raise FileNotFoundError(
            f"论文目录不存在：{paper_dir}"
        )

    if not paper_dir.is_dir():
        raise NotADirectoryError(
            f"路径不是目录：{paper_dir}"
        )

    pdf_paths = sorted(
        paper_dir.glob("*.pdf")
    )

    if not pdf_paths:
        raise FileNotFoundError(
            f"目录中没有 PDF 文件：{paper_dir}"
        )

    all_documents: list[Document] = []

    for pdf_path in pdf_paths:

        print(
            f"正在读取：{pdf_path.name}"
        )

        documents = load_pdf_pages(
            pdf_path
        )

        all_documents.extend(
            documents
        )

    return all_documents


def print_quality_summary(
    documents: list[Document],
) -> None:
    """
    Print a simple PDF extraction-quality report.

    Args:
        documents:
            Documents returned by load_pdf_directory().
    """
    empty_documents = [
        document
        for document in documents
        if document.metadata["is_empty"]
    ]

    warning_documents = [
        document
        for document in documents
        if document.metadata["parse_warnings"]
    ]

    print()
    print("=" * 70)
    print("PDF 解析质量汇总")
    print("=" * 70)

    print(
        f"总页数：{len(documents)}"
    )

    print(
        f"空文本页：{len(empty_documents)}"
    )

    print(
        "存在解析警告的页："
        f"{len(warning_documents)}"
    )

    if empty_documents:

        print()
        print("空文本页：")

        for document in empty_documents:

            metadata = document.metadata

            print(
                f"- {metadata['file_name']} "
                f"| PDF page "
                f"{metadata['page_number']}"
            )

    if warning_documents:

        print()
        print("解析警告：")

        for document in warning_documents:

            metadata = document.metadata

            print(
                f"- {metadata['file_name']} "
                f"| PDF page "
                f"{metadata['page_number']} "
                f"| "
                f"{metadata['parse_warnings']}"
            )


def print_document_sample(
    document: Document,
    preview_length: int = 800,
) -> None:
    """
    Print one Document sample.

    Args:
        document:
            LangChain Document to inspect.

        preview_length:
            Maximum number of page-content characters
            displayed in the terminal.
    """
    print()
    print("=" * 70)
    print("Document 示例")
    print("=" * 70)

    print()
    print("Document 类型：")
    print(type(document))

    print()
    print("metadata：")

    for key, value in document.metadata.items():
        print(
            f"{key}: {value}"
        )

    print()
    print("-" * 70)
    print("page_content：")
    print("-" * 70)

    preview = document.page_content[
        :preview_length
    ]

    print(preview)

    if len(document.page_content) > preview_length:
        print()
        print("......")


def main() -> None:
    """
    Run a local PDF loading test.

    This function is only executed when running:

        python document_loader.py
    """
    project_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    paper_dir = (
        project_root
        / "data"
        / "papers"
    )

    try:
        documents = load_pdf_directory(
            paper_dir
        )

        print_quality_summary(
            documents
        )

        if documents:
            print_document_sample(
                documents[0]
            )

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        pymupdf.FileDataError,
    ) as error:

        print()
        print(
            f"PDF 读取失败：{error}"
        )


if __name__ == "__main__":
    main()