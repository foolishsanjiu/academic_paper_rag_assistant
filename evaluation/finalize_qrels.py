"""Materialize the reviewed Chunk-level qrels.

The mapping below records the outcome of reviewing the semantic draft against
the indexed text.  Each inner list corresponds, in order, to one declared
source file/page.  The first Chunk is direct evidence (relevance 3); optional
following Chunks provide complementary evidence (relevance 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.build_annotation_pool import load_index_rows  # noqa: E402
from evaluation.dataset import load_questions  # noqa: E402
from evaluation.evaluate import write_json_atomic  # noqa: E402


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.json")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("qrels.json")


def _chunks(prefix: str, *indexes: int, page: int = 1) -> list[str]:
    return [f"{prefix}::page_{page}::chunk_{index}" for index in indexes]


AGSGAN = "94745577d08eeecb"
CG_DM = "839765ec83c5d156"
ATGAN = "35a65a0ca386b260"
AGGAN = "3a87dfa17edb9a9c"
BCNET = "b69d94be1b887bf1"
DIFFUSAR = "cf46cdbeb0170998"
LDGAN = "e59f7083701ab7a5"
AAE = "23cb1d819b8f4260"
NOISE = "37adac4f62176ddb"
SAR_DM = "3d3ee0ab76489bbe"
AZIMUTH_GAN = "9ea1895421b1ad48"
SHIP_GO = "f88cb45fe593fe8e"
LAND_COVER = "e1080d32ffe042fc"
STATISTICAL = "851b27fc272a4fe3"
FAGD = "5b120767fbe94ef9"


REVIEWED_CHUNKS: dict[str, list[list[str]]] = {
    "fact_001": [_chunks(AGSGAN, 2)],
    "fact_002": [_chunks(CG_DM, 2, 3)],
    "fact_003": [_chunks(ATGAN, 2)],
    "fact_004": [_chunks(AGGAN, 2)],
    "fact_005": [_chunks(BCNET, 2)],
    "fact_006": [_chunks(DIFFUSAR, 2)],
    "fact_007": [_chunks(LDGAN, 1, 2)],
    "fact_008": [_chunks(AAE, 2, 3)],
    "fact_009": [_chunks(NOISE, 1)],
    "fact_010": [_chunks(STATISTICAL, 2, 3)],
    "comparison_001": [_chunks(ATGAN, 2), _chunks(AZIMUTH_GAN, 2)],
    "comparison_002": [_chunks(AGSGAN, 2), _chunks(DIFFUSAR, 2)],
    "comparison_003": [_chunks(AGGAN, 2), _chunks(AAE, 1, 2)],
    "comparison_004": [_chunks(CG_DM, 2), _chunks(SHIP_GO, 3)],
    "comparison_005": [_chunks(SAR_DM, 1, 2), _chunks(DIFFUSAR, 2, 3)],
    "cross_001": [_chunks(CG_DM, 1), _chunks(DIFFUSAR, 1)],
    "cross_002": [
        _chunks(AGSGAN, 2),
        _chunks(AZIMUTH_GAN, 1, 2),
        _chunks(FAGD, 2),
    ],
    "cross_003": [
        _chunks(AGGAN, 2, 3),
        _chunks(BCNET, 1, 3),
        _chunks(SHIP_GO, 3, 4),
    ],
    "cross_004": [_chunks(CG_DM, 2), _chunks(SHIP_GO, 3), _chunks(LAND_COVER, 1)],
    "cross_005": [_chunks(NOISE, 1), _chunks(FAGD, 2, 3)],
    "fact_011": [_chunks(AGSGAN, 1, 2)],
    "fact_012": [_chunks(AGSGAN, 2)],
    "fact_013": [_chunks(CG_DM, 1, 2)],
    "fact_014": [_chunks(CG_DM, 3, 4)],
    "fact_015": [_chunks(ATGAN, 1, 2)],
    "fact_016": [_chunks(ATGAN, 1, page=16)],
    "fact_017": [_chunks(AGGAN, 2)],
    "fact_018": [_chunks(AGGAN, 4)],
    "fact_019": [_chunks(BCNET, 1)],
    "fact_020": [_chunks(BCNET, 3)],
    "fact_021": [_chunks(DIFFUSAR, 2)],
    "fact_022": [_chunks(DIFFUSAR, 3)],
    "fact_023": [_chunks(LDGAN, 2)],
    "fact_024": [_chunks(LDGAN, 3)],
    "fact_025": [_chunks(AAE, 1, 2)],
    "fact_026": [_chunks(AAE, 3)],
    "fact_027": [_chunks(NOISE, 1)],
    "fact_028": [_chunks(NOISE, 1)],
    "fact_029": [_chunks(SAR_DM, 1)],
    "fact_030": [_chunks(SAR_DM, 2)],
    "fact_031": [_chunks(AZIMUTH_GAN, 1, 2)],
    "fact_032": [_chunks(AZIMUTH_GAN, 1, 2, page=15)],
    "fact_033": [_chunks(SHIP_GO, 3)],
    "fact_034": [_chunks(SHIP_GO, 4)],
    "fact_035": [_chunks(LAND_COVER, 0, 1)],
    "fact_036": [_chunks(LAND_COVER, 1, 2)],
    "fact_037": [_chunks(STATISTICAL, 2, 3)],
    "fact_038": [_chunks(STATISTICAL, 1, 2)],
    "fact_039": [_chunks(FAGD, 2)],
    "fact_040": [_chunks(FAGD, 3)],
    "comparison_006": [_chunks(AGSGAN, 1, 2), _chunks(ATGAN, 1, 2)],
    "comparison_007": [_chunks(ATGAN, 1, 2), _chunks(AZIMUTH_GAN, 1, 2)],
    "comparison_008": [_chunks(CG_DM, 2), _chunks(SHIP_GO, 3)],
    "comparison_009": [_chunks(SAR_DM, 1, 2), _chunks(DIFFUSAR, 2)],
    "comparison_010": [_chunks(AGGAN, 2), _chunks(AAE, 1, 2)],
    "comparison_011": [_chunks(BCNET, 1), _chunks(LAND_COVER, 0, 1)],
    "comparison_012": [_chunks(LDGAN, 1, 2), _chunks(AGGAN, 2)],
    "comparison_013": [_chunks(NOISE, 0, 1), _chunks(STATISTICAL, 2, 3)],
    "comparison_014": [_chunks(FAGD, 2), _chunks(CG_DM, 1, 2)],
    "comparison_015": [_chunks(SHIP_GO, 3, 4), _chunks(BCNET, 1, 3)],
    "comparison_016": [_chunks(LAND_COVER, 0, 1), _chunks(SHIP_GO, 3)],
    "comparison_017": [_chunks(AAE, 2, 3), _chunks(AZIMUTH_GAN, 1, 2)],
    "comparison_018": [_chunks(AGSGAN, 1, 2), _chunks(FAGD, 2, 3)],
    "comparison_019": [_chunks(DIFFUSAR, 1, 2), _chunks(NOISE, 0, 1)],
    "comparison_020": [_chunks(SAR_DM, 1, 2), _chunks(STATISTICAL, 2, 3)],
    "cross_006": [
        _chunks(AGSGAN, 1, 2),
        _chunks(ATGAN, 1, 2),
        _chunks(AZIMUTH_GAN, 1, 2),
    ],
    "cross_007": [_chunks(CG_DM, 1, 2), _chunks(SHIP_GO, 3), _chunks(FAGD, 2)],
    "cross_008": [_chunks(AGGAN, 2, 3), _chunks(AAE, 2, 3), _chunks(CG_DM, 2)],
    "cross_009": [_chunks(BCNET, 1), _chunks(NOISE, 1), _chunks(ATGAN, 1, 2)],
    "cross_010": [_chunks(LDGAN, 1, 2), _chunks(CG_DM, 2), _chunks(SHIP_GO, 3)],
    "cross_011": [
        _chunks(AGSGAN, 1, 2),
        _chunks(DIFFUSAR, 2),
        _chunks(STATISTICAL, 2, 3),
    ],
    "cross_012": [_chunks(AGGAN, 2), _chunks(AAE, 1, 2), _chunks(LDGAN, 1, 2)],
    "cross_013": [_chunks(BCNET, 1), _chunks(SHIP_GO, 3), _chunks(LAND_COVER, 0, 1)],
    "cross_014": [_chunks(SAR_DM, 1, 2), _chunks(DIFFUSAR, 2), _chunks(NOISE, 0, 1)],
    "cross_015": [_chunks(FAGD, 2, 3), _chunks(NOISE, 0, 1), _chunks(CG_DM, 0, 2)],
    "cross_016": [_chunks(ATGAN, 1, 2), _chunks(AZIMUTH_GAN, 2), _chunks(AAE, 1, 2)],
    "cross_017": [_chunks(ATGAN, 1, 2), _chunks(LDGAN, 1, 2), _chunks(BCNET, 1)],
    "cross_018": [_chunks(CG_DM, 2, 4), _chunks(SHIP_GO, 3, 4), _chunks(LAND_COVER, 2, 7)],
    "cross_019": [_chunks(STATISTICAL, 2, 3), _chunks(CG_DM, 1, 2), _chunks(FAGD, 2)],
    "cross_020": [_chunks(CG_DM, 0, 1, 2), _chunks(STATISTICAL, 1), _chunks(LAND_COVER, 0, 1)],
}


def build_reviewed_qrels(
    questions: list[dict[str, Any]],
    index_rows: dict[str, dict[str, Any]],
    reviewed_chunks: dict[str, list[list[str]]],
) -> dict[str, list[dict[str, Any]]]:
    """Cross-check reviewed IDs against source declarations and index data."""
    answerable_ids = {
        question["id"] for question in questions if question["type"] != "no_answer"
    }
    if set(reviewed_chunks) != answerable_ids:
        missing = sorted(answerable_ids - set(reviewed_chunks))
        extra = sorted(set(reviewed_chunks) - answerable_ids)
        raise ValueError(f"reviewed_chunks 覆盖不完整：missing={missing}, extra={extra}")

    qrels: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        question_id = question["id"]
        if question["type"] == "no_answer":
            qrels[question_id] = []
            continue

        source_groups = reviewed_chunks[question_id]
        expected_pairs = list(zip(question["source_files"], question["source_pages"]))
        if len(source_groups) != len(expected_pairs):
            raise ValueError(f"{question_id} 的证据组数量与来源数量不一致。")

        judgments: list[dict[str, Any]] = []
        seen: set[str] = set()
        for (file_name, page_number), chunk_ids in zip(expected_pairs, source_groups):
            if not chunk_ids:
                raise ValueError(f"{question_id} 包含空证据组。")
            for position, chunk_id in enumerate(chunk_ids):
                if chunk_id in seen:
                    raise ValueError(f"{question_id} 包含重复 Chunk：{chunk_id}")
                seen.add(chunk_id)
                row = index_rows.get(chunk_id)
                if row is None:
                    raise ValueError(f"索引中不存在 reviewed Chunk：{chunk_id}")
                if (row["file_name"], int(row["page_number"])) != (
                    file_name,
                    page_number,
                ):
                    raise ValueError(f"{question_id} 的 reviewed Chunk 来源不匹配：{chunk_id}")
                direct = position == 0
                judgments.append(
                    {
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "page_number": page_number,
                        "relevance": 3 if direct else 2,
                        "reason": (
                            "Directly supports the expected answer for this source."
                            if direct
                            else "Provides complementary evidence for the expected answer."
                        ),
                    }
                )
        qrels[question_id] = judgments
    return qrels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the reviewed qrels file.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    qrels = build_reviewed_qrels(questions, load_index_rows(), REVIEWED_CHUNKS)
    write_json_atomic(args.output, qrels)
    print(
        f"Reviewed qrels written: questions={len(qrels)}, "
        f"judgments={sum(len(items) for items in qrels.values())}, "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
