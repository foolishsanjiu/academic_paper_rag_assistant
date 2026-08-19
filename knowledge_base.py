"""Knowledge-base metadata utilities."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_PAPER_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "papers"
)


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