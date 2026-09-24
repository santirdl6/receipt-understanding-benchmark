from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from receipt_benchmark.adapters import CordAdapter

app = typer.Typer(help="Receipt Understanding Benchmark utilities.")


def _default_mapping() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "cord_mapping.yaml"


DEFAULT_MAPPING = _default_mapping()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise typer.BadParameter("The input JSON must contain one object")
    return value


@app.command("inspect-cord")
def inspect_cord(
    annotation: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    mapping: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_MAPPING,
) -> None:
    adapter = CordAdapter.from_yaml(mapping)
    document = adapter.convert(_read_json(annotation), document_id=annotation.stem)
    source_counts = Counter(span.source_label or "<none>" for span in document.spans)
    mapped = sum(span.canonical_label is not None for span in document.spans)
    typer.echo(f"Document: {document.document_id}")
    typer.echo(f"Words: {len(document.spans)}")
    typer.echo(f"Mapped words: {mapped}")
    typer.echo("Categories:")
    for label, count in sorted(source_counts.items()):
        typer.echo(f"  {label}: {count}")


@app.command("convert-cord")
def convert_cord(
    annotation: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    document_id: Annotated[str | None, typer.Option()] = None,
    image_path: Annotated[str | None, typer.Option()] = None,
    split: Annotated[str | None, typer.Option()] = None,
    mapping: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_MAPPING,
) -> None:
    adapter = CordAdapter.from_yaml(mapping)
    document = adapter.convert(
        _read_json(annotation),
        document_id=document_id or annotation.stem,
        image_path=image_path,
        split=split,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Converted {len(document.spans)} words -> {output}")


if __name__ == "__main__":
    app()
