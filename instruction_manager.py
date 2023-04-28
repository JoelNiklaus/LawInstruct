import json
import random
from typing import Optional


class InstructionManager:
    """Class for managing instructions for different datasets."""
    def __init__(
            self,
            mode: str,
            instruction_bank_size: int,
            random_state: Optional[int] = None,
    ) -> None:
        if mode == 'english':
            json_file = 'instruction_prompts/en.json'
        elif mode == 'multilingual':
            json_file = 'instruction_prompts/multilingual.json'
        elif mode == 'dummy':
            json_file = 'instruction_prompts/dummy.json'
        else:
            raise ValueError(
                f'Mode {mode} should be either "english" or "multilingual"')

        self._random = random.Random(random_state or 1337)
        self._instruction_bank_size = instruction_bank_size
        self._instructions: dict[str, list[str]] = {}
        with open(json_file) as f:
            self._instructions = json.load(f)
        if not self._instructions or not all(self._instructions.values()):
            raise ValueError(
                f'Instruction bank {json_file} is empty or malformed.')

    def sample(self, task_type: str) -> str:
        """Sample an instruction from the bank."""
        instruction_bank = self._instructions[task_type]
        restricted_by_size = instruction_bank[:self._instruction_bank_size]
        # Make sure we didn't mess up the math...
        assert len(restricted_by_size) <= self._instruction_bank_size
        return self._random.choice(restricted_by_size)
