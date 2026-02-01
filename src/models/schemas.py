"""
Schemas Pydantic para validação da estrutura de saída JSON
"""
from typing import List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class BlockType(str, Enum):
    """Tipos de blocos de conteúdo"""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    IMAGE = "image"


class BBox(BaseModel):
    """Bounding box normalizado (0-1)"""
    x1: float = Field(ge=0.0, le=1.0, description="Coordenada X superior esquerda")
    y1: float = Field(ge=0.0, le=1.0, description="Coordenada Y superior esquerda")
    x2: float = Field(ge=0.0, le=1.0, description="Coordenada X inferior direita")
    y2: float = Field(ge=0.0, le=1.0, description="Coordenada Y inferior direita")

    def to_list(self) -> List[float]:
        """Retorna bbox como lista [x1, y1, x2, y2]"""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BBox':
        """Cria BBox a partir de lista [x1, y1, x2, y2]"""
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])

    @classmethod
    def from_absolute(cls, bbox: List[float], page_width: float, page_height: float) -> 'BBox':
        """Converte coordenadas absolutas para relativas (0-1)"""
        return cls(
            x1=bbox[0] / page_width,
            y1=bbox[1] / page_height,
            x2=bbox[2] / page_width,
            y2=bbox[3] / page_height
        )


class Block(BaseModel):
    """Bloco de conteúdo (parágrafo, tabela, etc)"""
    block_id: str = Field(description="ID único do bloco (ex: p1_b1)")
    type: BlockType = Field(description="Tipo do bloco")
    text: Optional[str] = Field(default=None, description="Texto do bloco (se aplicável)")
    bbox: List[float] = Field(description="Bounding box [x1, y1, x2, y2] normalizado")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0, description="Confiança da extração")
    rows: Optional[List[List[str]]] = Field(default=None, description="Linhas da tabela (se type=table)")
    
    class Config:
        use_enum_values = True


class Page(BaseModel):
    """Página do documento"""
    page: int = Field(ge=1, description="Número da página")
    source: Literal["digital", "ocr"] = Field(description="Método de extração")
    blocks: List[Block] = Field(default_factory=list, description="Blocos de conteúdo da página")
    width: Optional[float] = Field(default=None, description="Largura da página (pts)")
    height: Optional[float] = Field(default=None, description="Altura da página (pts)")


class Document(BaseModel):
    """Documento completo processado"""
    doc_id: str = Field(description="ID do documento")
    source_file: str = Field(description="Nome do arquivo PDF original")
    total_pages: int = Field(ge=1, description="Total de páginas")
    processing_date: datetime = Field(default_factory=datetime.now, description="Data de processamento")
    pages: List[Page] = Field(default_factory=list, description="Páginas processadas")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_json_dict(self) -> dict:
        """Exporta para dicionário JSON serializable"""
        return self.model_dump(mode='json')
