import json
from pathlib import Path

import pytest

from receipt_benchmark.adapters import CordAdapter, CordAdapterError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cord_sample.json"
MAPPING = ROOT / "configs" / "cord_mapping.yaml"


def test_converts_valid_lines_and_preserves_groups() -> None:
    adapter = CordAdapter.from_yaml(MAPPING)
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))

    result = adapter.convert(record, document_id="cord_001", split="train")

    assert result.source_dataset == "CORD"
    assert result.image.width == 1000
    assert len(result.spans) == 2
    assert result.spans[0].canonical_label == "ITEM_DESCRIPTION"
    assert result.spans[0].group_id == "1"
    assert result.spans[0].bbox == (100.0, 200.0, 250.0, 230.0)
    assert result.spans[1].canonical_label == "TOTAL_AMOUNT"
    assert result.source_parse["total"]["total_price"] == "5.00"


def test_accepts_hugging_face_serialized_ground_truth() -> None:
    adapter = CordAdapter.from_yaml(MAPPING)
    annotation = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = adapter.convert(
        {"ground_truth": json.dumps(annotation)}, document_id="cord_hf_001"
    )
    assert len(result.spans) == 2


def test_rejects_missing_valid_line() -> None:
    adapter = CordAdapter.from_yaml(MAPPING)
    with pytest.raises(CordAdapterError, match="valid_line"):
        adapter.convert({}, document_id="broken")

