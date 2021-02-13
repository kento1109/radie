from typing import Dict, List, Union

from pydantic import BaseModel, Field

class Tagger(BaseModel):
    tokens: List[str]
    labels: List[str]


class Concept(BaseModel):
    """
    extracted observation and clinical findings,
    and whose certainty scale
    """
    clinical_concept: str
    certainty_scale: str

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


class Statement(BaseModel):
    """確信度判定モデルの必要な情報を保持する"""
    tokens: List[str]
    obj: Entity


class RelationStatement(Statement):
    """関係抽出モデルの必要な情報を保持する"""
    attr: Entity


class OAModel(BaseModel):
    """構造化結果をObject-Attributeの形式を保持するクラス"""
    findings_seq: int
    obj_entity: Entity
    attr_entity: Entity
    relation_score: float

    def chunking(self):
        self.obj_entity.chunking()
        self.attr_entity.chunking()


class OAVTripletModel(BaseModel):
    """構造化結果をOAVTripletの形式で保持するクラス"""
    findings_seq: int
    obj_tokens: List[str]
    attr_tokens: List[str]
    value_entity: str
    relation_score: float

    def chunking(self):
        self.obj_tokens = ''.join(self.obj_tokens)
        self.attr_tokens = ''.join(self.attr_tokens)


class Structured_Report(BaseModel):
    structured_data_list: Union[List[OAModel], List[OAVTripletModel]]

    def __iter__(self):
        return iter(self.structured_data_list)

    def __getitem__(self, i):
        return self.structured_data_list[i]

    def __len__(self):
        return len(self.structured_data_list)

    def chunking(self):
        if self.structured_data_list:
            for structured_data in self.structured_data_list:
                structured_data.chunking()


class NormMap(BaseModel):
    """store norm infomation for entity mapping"""
    start_idx: int
    normed: List[str]