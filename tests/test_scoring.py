import pytest

from receipt_benchmark.scoring import score_run


@pytest.mark.parametrize(
    ("reference_unit_price", "prediction_unit_price", "expected_f1"),
    [
        ("@ 12.000", "@12.000", 1.0),
        ("@ 12.000", "12,000", 1.0),
        ("@ 12.000", "@ 24.000", 0.0),
        (None, "24.000", 0.0),
    ],
)
def test_unit_price_marker_in_cord_scoring(
    reference_unit_price: str | None,
    prediction_unit_price: str,
    expected_f1: float,
) -> None:
    line = {
        "ITEM_DESCRIPTION": "Arem Arem",
        "ITEM_QUANTITY": "2 x",
        "ITEM_LINE_TOTAL": "24.000",
    }
    report = score_run([
        {
            "model": "test",
            "split": "validation",
            "json_valid": True,
            "reference": {"items": [{**line, "ITEM_UNIT_PRICE": reference_unit_price}]},
            "prediction": {"items": [{**line, "ITEM_UNIT_PRICE": prediction_unit_price}]},
            "latency_ms": 1.0,
            "peak_vram_bytes": 0,
        },
    ])

    assert report["fields"]["ITEM_UNIT_PRICE"]["f1"] == expected_f1
    assert report["line_f1"] == expected_f1
