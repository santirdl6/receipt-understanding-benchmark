from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


class ImageMetadata(BaseModel): #Representa información de la imagen
    path: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class TextSpan(BaseModel): #Representa una palabra detectada o anotada
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    bbox: BBox
    polygon: list[Point]
    reading_order: int = Field(ge=0)
    source_label: str | None = None
    canonical_label: str | None = None
    group_id: str | None = None
    sub_group_id: str | None = None
    row_id: str | None = None
    is_key: bool | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> TextSpan:
        x1, y1, x2, y2 = self.bbox
        if x1 > x2 or y1 > y2:
            raise ValueError("bbox must satisfy x1 <= x2 and y1 <= y2")
        if len(self.polygon) < 4:
            raise ValueError("polygon must contain at least four points")
        return self


class CanonicalDocument(BaseModel): #Representa un ticket completo
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    document_id: str
    source_dataset: str
    source_split: str | None = None
    language: list[str] = Field(default_factory=list)
    country: str | None = None
    currency: str | None = None
    image: ImageMetadata = Field(default_factory=ImageMetadata)
    spans: list[TextSpan] = Field(default_factory=list)
    source_parse: dict | list | None = None
