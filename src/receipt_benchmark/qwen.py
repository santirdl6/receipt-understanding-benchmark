"""Run the same image-to-JSON benchmark with either generalist Qwen checkpoint."""

from __future__ import annotations

import json
from io import BytesIO
from itertools import chain, islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterator, Literal

from receipt_benchmark.scoring import ReceiptExtraction, cord_reference, parse_prediction

if TYPE_CHECKING:
    from PIL.Image import Image

QwenModel = Literal["4b", "8b"]
CordSplit = Literal["validation", "test"]
PROMPT = """Extract the line items and grand total from this receipt image.
Return only one JSON object with this structure:
{"items":[{"ITEM_DESCRIPTION":null,"ITEM_QUANTITY":null,"ITEM_UNIT_PRICE":null,
"ITEM_LINE_TOTAL":null}],"TOTAL_AMOUNT":null}
Use strings copied from the receipt for visible values and null for missing values.
Keep the order of the main product lines, omitting modifiers and voided lines.
Do not calculate missing amounts. Use an empty items list if no items are visible."""


class QwenRunner:
    def __init__(
        self, model: QwenModel, device: str, max_pixels: int, max_new_tokens: int,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("This benchmark requires a CUDA GPU.")
        with torch.cuda.device(self.device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("This benchmark requires a GPU with native BF16 support.")

        self.model_id = f"Qwen/Qwen3-VL-{model.upper()}-Instruct"
        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, revision=revision, dtype=torch.bfloat16,
            device_map={"": str(self.device)}, attn_implementation="sdpa",
        ).eval()
        self.revision = self.model.config._commit_hash or revision
        self.processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.processor.image_processor.size = {
            **self.processor.image_processor.size,
            "longest_edge": max_pixels,
        }

    def predict(self, image: Image) -> tuple[str, float, int]:
        cuda = self.torch.cuda
        cuda.synchronize(self.device)
        cuda.reset_peak_memory_stats(self.device)
        start = perf_counter()
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
            )
        completion = generated[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            completion, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        cuda.synchronize(self.device)
        return raw, (perf_counter() - start) * 1000, cuda.max_memory_allocated(self.device)


def _cord_rows(cord_dir: Path, split: CordSplit) -> Iterator[dict[str, Any]]:
    from pyarrow.parquet import ParquetFile

    files = sorted((cord_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise ValueError(f"No {split} Parquet files under {cord_dir / 'data'}.")
    for path in files:
        with ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(
                batch_size=8, columns=["image", "ground_truth"],
            ):
                yield from batch.to_pylist()


def _image(row: dict[str, Any]) -> Image:
    from PIL import Image

    with Image.open(BytesIO(row["image"]["bytes"])) as image:
        return image.convert("RGB")


def run_cord(
    cord_dir: Path,
    output: Path,
    *,
    model: QwenModel,
    split: CordSplit,
    limit: int | None,
    device: str,
    max_pixels: int,
    max_new_tokens: int,
    revision: str | None = None,
) -> int:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    rows = iter(islice(_cord_rows(cord_dir, split), limit))
    first = next(rows, None)
    if first is None:
        raise ValueError(f"The {split} split is empty.")

    runner = QwenRunner(model, device, max_pixels, max_new_tokens, revision)
    runner.predict(_image(first))  # Warm up outside the measured records.
    metadata = {
        "dataset": "naver-clova-ix/cord-v2",
        "split": split,
        "model": runner.model_id,
        "revision": runner.revision,
        "gpu": runner.torch.cuda.get_device_name(runner.device),
        "dtype": "bfloat16",
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "prompt": PROMPT,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            for count, row in enumerate(chain([first], rows), start=1):
                raw, latency_ms, peak_vram_bytes = runner.predict(_image(row))
                prediction, json_valid = parse_prediction(raw)
                reference = cord_reference(row["ground_truth"])
                record = {
                    **metadata,
                    "document_id": f"cord_v2_{split}_{count - 1:06d}",
                    "raw_output": raw,
                    "json_valid": json_valid,
                    "prediction": _fields(prediction),
                    "reference": _fields(reference),
                    "latency_ms": latency_ms,
                    "peak_vram_bytes": peak_vram_bytes,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.close()
            output.hardlink_to(temporary)
        finally:
            temporary.unlink(missing_ok=True)
    return count


def _fields(extraction: ReceiptExtraction | None) -> dict | None:
    return extraction.model_dump(by_alias=True) if extraction is not None else None
