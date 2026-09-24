from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from receipt_benchmark.schema import CanonicalDocument, ImageMetadata, TextSpan


class CordAdapterError(ValueError):
    """Raised when a CORD annotation cannot be converted safely."""


def _as_dict(record: dict[str, Any]) -> dict[str, Any]:
    """Unwrap the serialized `ground_truth` used by HF CORD exports."""
    ground_truth = record.get("ground_truth")
    if ground_truth is None:
        return record
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            decoded = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise CordAdapterError("ground_truth is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise CordAdapterError("ground_truth JSON must contain an object")
        return decoded
    raise CordAdapterError("ground_truth must be an object or a JSON string")


def _point(raw: dict[str, Any]) -> tuple[float, float]:
    try:
        return float(raw["x"]), float(raw["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CordAdapterError(f"Invalid quadrilateral point: {raw!r}") from exc


def _polygon(quad: dict[str, Any] | list[Any]) -> list[tuple[float, float]]:
    if isinstance(quad, list):
        points = [_point(item) for item in quad]
    elif isinstance(quad, dict):
        ordered_keys = ("x1", "x2", "x3", "x4")
        if all(key in quad for key in ordered_keys):
            points = [_point(quad[key]) for key in ordered_keys]
        else:
            numeric_items = [value for value in quad.values() if isinstance(value, dict)]
            points = [_point(item) for item in numeric_items]
    else:
        raise CordAdapterError(f"quad must be an object or list, got {type(quad).__name__}")
    if len(points) < 4:
        raise CordAdapterError(f"quad has {len(points)} points; expected at least four")
    return points


def _bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


class CordAdapter:
    def __init__(self, mapping: dict[str, str | None]):
        self.mapping = mapping

    @classmethod
    def from_yaml(cls, path: str | Path) -> CordAdapter:
        with Path(path).open(encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle)
        if not isinstance(mapping, dict):
            raise CordAdapterError("CORD mapping must be a YAML object")
        return cls(mapping=mapping)

    def convert(
        self,
        record: dict[str, Any],
        *,
        document_id: str,
        image_path: str | None = None,
        split: str | None = None,
    ) -> CanonicalDocument:
        annotation = _as_dict(record)
        valid_lines = annotation.get("valid_line")
        if not isinstance(valid_lines, list):
            raise CordAdapterError("CORD annotation does not contain a valid_line list")

        meta = annotation.get("meta") or {}
        image_size = meta.get("image_size") or record.get("image_size") or {}
        width = image_size.get("width")
        height = image_size.get("height")
        spans: list[TextSpan] = []

        for line_index, line in enumerate(valid_lines):
            if not isinstance(line, dict):
                raise CordAdapterError(f"valid_line[{line_index}] is not an object")
            category = line.get("category")
            canonical = self.mapping.get(category) if category else None
            words = line.get("words") or []
            for word_index, word in enumerate(words):
                if not isinstance(word, dict):
                    raise CordAdapterError(
                        f"valid_line[{line_index}].words[{word_index}] is not an object"
                    )
                text = word.get("text")
                if not isinstance(text, str):
                    raise CordAdapterError(
                        f"valid_line[{line_index}].words[{word_index}] has no text"
                    )
                points = _polygon(word.get("quad"))
                spans.append(
                    TextSpan(
                        id=f"word_{len(spans):06d}",
                        text=text,
                        bbox=_bbox(points),
                        polygon=points,
                        reading_order=len(spans),
                        source_label=category,
                        canonical_label=canonical,
                        group_id=_optional_id(line.get("group_id")),
                        sub_group_id=_optional_id(line.get("sub_group_id")),
                        row_id=_optional_id(line.get("row_id")),
                        is_key=line.get("is_key"),
                    )
                )

        return CanonicalDocument(
            document_id=document_id,
            source_dataset="CORD",
            source_split=split,
            language=["id"],
            country="ID",
            image=ImageMetadata(path=image_path, width=width, height=height),
            spans=spans,
            source_parse=annotation.get("gt_parse"),
        )


def _optional_id(value: Any) -> str | None:
    return None if value is None else str(value)
