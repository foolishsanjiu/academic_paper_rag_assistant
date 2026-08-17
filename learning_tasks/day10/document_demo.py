"""Inspect a complete LangChain Document."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from document_loader import load_pdf_directory


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    paper_dir = (
        project_root
        / "data"
        / "papers"
    )

    documents = load_pdf_directory(
        paper_dir
    )

    if not documents:
        print("没有读取到 Document。")
        return

    document = documents[0]

    print("=" * 70)
    print("Document 类型")
    print("=" * 70)

    print(type(document))

    print()
    print("=" * 70)
    print("完整 Document")
    print("=" * 70)

    print(document)

    print()
    print("=" * 70)
    print("page_content")
    print("=" * 70)

    print(document.page_content[:1000])

    print()
    print("=" * 70)
    print("metadata")
    print("=" * 70)

    for key, value in document.metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()