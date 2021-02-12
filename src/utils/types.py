from typing import Dict
from typing import List

from pydantic import BaseModel

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
