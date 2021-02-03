from typing import Dict
from typing import List

from pydantic import BaseModel

class Tagger(BaseModel):
    tokens: List[str]
    labels: List[str]
