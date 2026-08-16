"""Compare PyMuPDF text extraction order."""

from pathlib import Path

import pymupdf


PDF_PATH = Path(
    "data/papers/Exploring_a_Novel_Content-Guided_High-Resolution_SAR_Ship_Image_Generation_Method.pdf"
)

PAGE_INDEX = 1


def main() -> None:
    with pymupdf.open(PDF_PATH) as document:
        page = document[PAGE_INDEX]

        original_order = page.get_text(
            "text",
            sort=False,
        )

        sorted_order = page.get_text(
            "text",
            sort=True,
        )

    print("=" * 70)
    print("sort=False")
    print("=" * 70)
    print(original_order[:3000])

    print()

    print("=" * 70)
    print("sort=True")
    print("=" * 70)
    print(sorted_order[:3000])


if __name__ == "__main__":
    main()