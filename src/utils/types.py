from typing import Dict, List, Union, Optional

from pydantic import BaseModel, Field

class Tagger(BaseModel):
    tokens: List[str]
    labels: List[str]


class Norm(BaseModel):
    """controlled vocabllary class"""
    # uid: str
    name: str


class Entity(BaseModel):
    name: str = Field(name="entity label name such as Imaging_observation")
    tokens: List[str] = Field(name="mention strings")
    start_idx: int = Field(name="start index of tokens",
                           description="used for specifying entity")
    norms: List[Norm] = None
    

    def chunking(self):
        self.tokens = ''.join(self.tokens)

class Object(BaseModel):
    """
    extracted observation and clinical findings,
    and whose certainty scale
    """
    entity: Entity
    certainty_scale: str


class Statement(BaseModel):
    """確信度判定モデルの必要な情報を保持する"""
    tokens: List[str]
    obj: Entity


class RelationStatement(Statement):
    """関係抽出モデルの必要な情報を保持する"""
    attr: Entity


class StructuredModel(BaseModel):
    clinical_object: Object
    attributes: List[Optional[Entity]]

    def chunking(self):
        self.clinical_object.entity.chunking()
        for attribute in self.attributes:
            attribute.chunking()


class NormMap(BaseModel):
    """store norm infomation for entity mapping"""
    start_idx: int
    normed: List[str]