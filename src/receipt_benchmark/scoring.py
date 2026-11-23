"""Field-level results for the CORD v2 image-to-text pilot."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from statistics import median

from pydantic import BaseModel, ConfigDict, Field, ValidationError

LINE_FIELDS = {
    "ITEM_DESCRIPTION": "nm",
    "ITEM_QUANTITY": "cnt",
    "ITEM_UNIT_PRICE": "unitprice",
    "ITEM_LINE_TOTAL": "price",
}
TOTAL = "TOTAL_AMOUNT"


class ItemExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    description: str | None = Field(default=None, alias="ITEM_DESCRIPTION")
    quantity: str | None = Field(default=None, alias="ITEM_QUANTITY")
    unit_price: str | None = Field(default=None, alias="ITEM_UNIT_PRICE")
    line_total: str | None = Field(default=None, alias="ITEM_LINE_TOTAL")


class ReceiptExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    items: list[ItemExtraction]
    total: str | None = Field(default=None, alias=TOTAL)


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_prediction(raw: str) -> tuple[ReceiptExtraction | None, bool]:
    try:
        value = json.loads(raw, parse_constant=_invalid_constant)
    except ValueError:
        return None, False
    try:
        return ReceiptExtraction.model_validate(value), True
    except ValidationError:
        return None, True


def cord_reference(ground_truth: str) -> ReceiptExtraction:
    source = json.loads(ground_truth)["gt_parse"]
    items = source.get("menu", [])
    if isinstance(items, dict):
        items = [items]
    return ReceiptExtraction.model_validate({
        "items": [
            {field: item.get(key) for field, key in LINE_FIELDS.items()}
            for item in items
        ],
        TOTAL: source.get("total", {}).get("total_price"),
    })


def _normalize(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    text = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if field == "ITEM_DESCRIPTION":
        return text or None

    if field == "ITEM_QUANTITY":
        numeric = re.sub(r"\s*[x×]$", "", text)
    else:
        # Receipts can mark an explicit unit price with a leading @.
        amount = text.removeprefix("@").lstrip() if field == "ITEM_UNIT_PRICE" else text
        numeric = re.sub(r"^(?:idr|rp\.?)\s*", "", amount).replace(" ", "")
        # CORD uses both 75,000 and 75.000 for grouped rupiah amounts.
        if re.fullmatch(r"[+-]?\d{1,3}(?:[,\.]\d{3})+(?:[,\.]\d{1,2})?", numeric):
            integer, fraction = re.split(r"[,\.](?=\d+$)", numeric, maxsplit=1)
            integer = integer.replace(",", "").replace(".", "")
            numeric = integer + (fraction if len(fraction) == 3 else "." + fraction)
    numeric = numeric.replace(",", ".")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric):
        return str(Decimal(numeric).normalize())
    return text or None


def _fields(receipt: dict | None) -> dict[str, Counter]:
    fields = {key: Counter() for key in (*LINE_FIELDS, TOTAL)}
    if receipt is None:
        return fields

    for item in receipt["items"]:
        name = _normalize(item.get("ITEM_DESCRIPTION"), "ITEM_DESCRIPTION")
        for key in LINE_FIELDS:
            value = _normalize(item.get(key), key)
            if value is not None:
                fields[key][(name, value)] += 1
    total = _normalize(receipt.get(TOTAL), TOTAL)
    if total is not None:
        fields[TOTAL][total] += 1
    return fields


def _lines(receipt: dict | None) -> Counter:
    items = receipt["items"] if receipt is not None else []
    return Counter(
        tuple(_normalize(item.get(key), key) for key in LINE_FIELDS)
        for item in items
    )


def score_run(records: list[dict]) -> dict:
    if not records:
        raise ValueError("The predictions file is empty.")
    keys = (*LINE_FIELDS, TOTAL)
    counts = {key: Counter() for key in keys}
    lines = Counter()
    for row in records:
        expected_lines = _lines(row["reference"])
        predicted_lines = _lines(row["prediction"])
        lines["correct"] += sum((expected_lines & predicted_lines).values())
        lines["reference"] += sum(expected_lines.values())
        lines["predicted"] += sum(predicted_lines.values())

        expected = _fields(row["reference"])
        predicted = _fields(row["prediction"])
        for key in keys:
            counts[key]["correct"] += sum((expected[key] & predicted[key]).values())
            counts[key]["reference"] += sum(expected[key].values())
            counts[key]["predicted"] += sum(predicted[key].values())

    fields = {}
    for key, count in counts.items():
        denominator = count["reference"] + count["predicted"]
        fields[key] = {
            **count,
            "f1": 2 * count["correct"] / denominator if denominator else None,
        }
    line_denominator = lines["reference"] + lines["predicted"]
    latencies = sorted(row["latency_ms"] for row in records)
    return {
        "model": records[0]["model"],
        "split": records[0]["split"],
        "documents": len(records),
        "fields": fields,
        "line_f1": 2 * lines["correct"] / line_denominator if line_denominator else None,
        "total_accuracy": (
            counts[TOTAL]["correct"] / counts[TOTAL]["reference"]
            if counts[TOTAL]["reference"] else None
        ),
        "json_valid_rate": sum(row["json_valid"] for row in records) / len(records),
        "schema_valid_rate": sum(row["prediction"] is not None for row in records) / len(records),
        "latency_ms_median": median(latencies),
        "latency_ms_p95": latencies[(95 * len(latencies) + 99) // 100 - 1],
        "peak_vram_mib": max(row["peak_vram_bytes"] for row in records) / 2**20,
    }
