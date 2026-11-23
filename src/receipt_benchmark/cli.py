from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from receipt_benchmark.adapters import CordAdapter

app = typer.Typer(help="Receipt Understanding Benchmark utilities.")

#Proporciona comandos para utilizar proyecto desde la terminal

def _default_mapping() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "cord_mapping.yaml"


DEFAULT_MAPPING = _default_mapping()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise typer.BadParameter("The input JSON must contain one object")
    return value

#Inspecciona una anotación sin crear otro archivo
#Sirve para saber cuantas palabras hay, cuantas están mapeadas y cuantas categorías hay
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

#Convierte una anotación de CORD a nuestro esquema canónico y la guarda en un archivo JSON
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


class QwenSize(str, Enum):
    small = "4b"
    large = "8b"


class EvaluationSplit(str, Enum):
    validation = "validation"
    test = "test"


@app.command("run-qwen-cord")
def run_qwen_cord(
    cord_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Argument(dir_okay=False)],
    model: Annotated[QwenSize, typer.Option()] = QwenSize.large,
    split: Annotated[EvaluationSplit, typer.Option()] = EvaluationSplit.validation,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    device: Annotated[str, typer.Option()] = "cuda:0",
    max_pixels: Annotated[int, typer.Option(min=65_536)] = 1_310_720,
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 1024,
    revision: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Extract fields from local CORD v2 images with Qwen3-VL."""
    from receipt_benchmark.qwen import run_cord

    try:
        count = run_cord(
            cord_dir, output, model=model.value, split=split.value, limit=limit,
            device=device, max_pixels=max_pixels, max_new_tokens=max_new_tokens,
            revision=revision,
        )
    except ImportError as exc:
        raise typer.BadParameter("Install the runner with pip install -e '.[qwen]'.") from exc
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Saved {count} records to {output}")


@app.command("score-cord")
def score_cord(
    predictions: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Summarize extraction quality, latency and peak allocated VRAM for a JSONL run."""
    from receipt_benchmark.scoring import score_run

    with predictions.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    try:
        summary = score_run(records)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
