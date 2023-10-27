from abc import ABC, abstractmethod
from typing import Dict, List
class Model(ABC):

    @abstractmethod
    def generate(self, prompts: List[str], temp:float = 0.0):
        pass

    def generate_batch(self, prompts : List[str]):
        pass
