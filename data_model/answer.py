from pydantic import BaseModel
from typing import List, Tuple
class Answer(BaseModel):
    answer: str
    past_conversations: List[Tuple[str, str]]
    source: str