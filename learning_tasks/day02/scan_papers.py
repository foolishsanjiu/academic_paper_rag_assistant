from pathlib import Path
import json

def collect_pdf_info(paper_dir:Path) -> list[dict]:
    """
    Collects information about PDF files in the given directory.

    Args:
        paper_dir (Path): The directory containing PDF files.

    Returns:
        list[dict]: A list of dictionaries containing information about each PDF file.
    """
    if not paper_dir.exists():
        raise FileNotFoundError(f"论文目录不存在：{paper_dir}")

    if not paper_dir.is_dir():
        raise NotADirectoryError(f"提供的路径不是目录：{paper_dir}")

    papers : list[dict] = []

    for pdf_path in sorted(paper_dir.rglob("*.pdf")):
        file_size_bytes = pdf_path.stat().st_size
        paper_info = {
            "file_name": pdf_path.name,
            "relative_path": pdf_path.relative_to(paper_dir).as_posix(),
            "size_bytes": file_size_bytes,
            "size_mb": round(file_size_bytes / 1024 / 1024, 2),
        }
        papers.append(paper_info)
        
    return papers

def save_to_json(papers: list[dict], output_path: Path) -> None:
    """
    将论文信息保存为 JSON 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            papers,
            file,
            ensure_ascii=False,
            indent=2,
        )

def main() -> None:
    # scan_papers.py 位于：
    # 项目根目录/learning_tasks/day02/scan_papers.py
    project_root = Path(__file__).resolve().parents[2]

    paper_dir = project_root / "data" / "papers"
    output_path = project_root / "data" / "papers.json"

    papers = collect_pdf_info(paper_dir)
    save_to_json(papers, output_path)

    print(f"论文目录：{paper_dir}")
    print(f"共发现 {len(papers)} 篇 PDF")
    print(f"结果已保存至：{output_path}")

    for index, paper in enumerate(papers, start=1):
        print(
            f"{index}. {paper['file_name']} "
            f"({paper['size_mb']} MB)"
        )


if __name__ == "__main__":
    main()