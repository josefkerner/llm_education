from pydantic import BaseModel
from typing import List, Tuple

class Question(BaseModel):
    question: str
    context: List[Tuple[str, str]]