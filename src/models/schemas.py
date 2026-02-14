"""
Pydantic schemas for JSON output structure validation
"""
from typing import List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class BlockType(str, Enum):
    """Content block types"""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    IMAGE = "image"


class BBox(BaseModel):
    """Normalized bounding box (0-1)"""
    x1: float = Field(ge=0.0, le=1.0, description="Top-left X coordinate")
    y1: float = Field(ge=0.0, le=1.0, description="Top-left Y coordinate")
    x2: float = Field(ge=0.0, le=1.0, description="Bottom-right X coordinate")
    y2: float = Field(ge=0.0, le=1.0, description="Bottom-right Y coordinate")

    def to_list(self) -> List[float]:
        """Return bbox as list [x1, y1, x2, y2]"""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BBox':
        """Create BBox from list [x1, y1, x2, y2]"""
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    @classmethod
    def from_absolute(cls, bbox: List[float], page_width: float, page_height: float) -> 'BBox':
        """Convert absolute coordinates to relative (0-1)"""
        return cls(
            x1=bbox[0] / page_width,
            y1=bbox[1] / page_height,
            x2=bbox[2] / page_width,
            y2=bbox[3] / page_height
        )


class Block(BaseModel):
    """Content block (paragraph, table, etc.)"""
    block_id: str = Field(description="Unique block ID (e.g., p1_b1)")
    type: BlockType = Field(description="Block type")
    text: Optional[str] = Field(default=None, description="Block text (if applicable)")
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2] normalized")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Extraction confidence")
    rows: Optional[List[List[str]]] = Field(default=None, description="Table rows (if type=table)")
    lines: Optional[List[dict]] = Field(default=None, description="Individual lines with bbox: [{'text': str, 'bbox': [x1,y1,x2,y2]}]")

    class Config:
        use_enum_values = True


class Page(BaseModel):
    """Document page"""
    page: int = Field(ge=1, description="Page number")
    source: Literal["digital", "ocr"] = Field(description="Extraction method")
    blocks: List[Block] = Field(default_factory=list, description="Page content blocks")
    width: Optional[float] = Field(default=None, description="Page width (pts)")
    height: Optional[float] = Field(default=None, description="Page height (pts)")


class Document(BaseModel):
    """Complete processed document"""
    doc_id: str = Field(description="Document ID")
    source_file: str = Field(description="Original PDF filename")
    total_pages: int = Field(ge=1, description="Total pages")
    processing_date: datetime = Field(default_factory=datetime.now, description="Processing date")
    pages: List[Page] = Field(default_factory=list, description="Processed pages")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_json_dict(self) -> dict:
        """Export to JSON-serializable dictionary"""
        return self.model_dump(mode='json')
