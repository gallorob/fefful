import json
import os
from datetime import datetime
from typing import Dict, List, Tuple


class HistoryManager:
    def __init__(self,
                 where: str):
        self.where = where
        t = datetime.now()
        self.run_name = t.strftime('%Y%m%d%H%M%S')
        self._history: Dict[int, Tuple[bool, List[int]]] = {}

    def add_choices(self,
                    generation: int,
                    choices: List[int],
                    human: bool = True):
        assert generation not in self._history.keys(), f'Choices for generation {generation} already exist!'
        self._history[generation] = (human, choices)

    def has_choices(self,
                    generation: int) -> bool:
        return generation in self._history.keys()

    def get_choices(self,
                    generation: int) -> List[int]:
        assert generation in self._history.keys(), f'Choices for generation {generation} don\'t exist!'
        return self._history.get(generation, (True, []))

    def save(self):
        with open(os.path.join(self.where, f'{self.run_name}.history'), 'w') as f:
            json.dump(self._history, f)

    def load(self,
             filename: str):
        with open(os.path.join(self.where, f'{filename}.history'), 'r') as f:
            self._history = {int(k): v for k, v in json.load(f).items()}
