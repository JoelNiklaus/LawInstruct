import json
import random
import sys
from typing import NamedTuple, Optional

from absl import logging


class Instruction(NamedTuple):
    instruction: str
    lang: str


# TODO(arya): When off plane:
#   (1) install immutabledict
#   (2) use it to make InstructionManager._instructions hashable
#   (3) enable caching for this function.
# Caching this function makes sense because it will typically be
# called with the same arguments repeatedly (i.e., for one dataset) before the
# next arguments arise.
# @functools.lru_cache(maxsize=16)
def _get_all_lang_instructions(
        group: str,
        all_instructions: dict[str, dict[str, list[str]]],
        size_per_lang: int = sys.maxsize,
) -> list[tuple[str, str]]:
    """Combines individual-language instructions into one list."""
    result = [
        (instruction, lang)
        for lang, groups in all_instructions.items()
        for instruction in groups[group][:size_per_lang]
    ]
    # Make sure we didn't mess up the math.
    number_of_langs = len(all_instructions)
    assert len(result) <= size_per_lang * number_of_langs

    return result


class InstructionManager:
    """Class for managing instructions for different datasets."""

    def __init__(
            self,
            mode: str,
            instruction_bank_size: int,
            random_state: Optional[int] = 42,
    ) -> None:
        """Creates an instruction bank that can be sampled from.

        Args:
            mode: whether English-only or multlingual instructions
            instruction_bank_size: number of instructions per
                (language x task_type) pair
            random_state: To ensure reproducibility
        """
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
        # JSON file's structure is lang_code -> instruction_group -> text.
        with open(json_file, encoding='utf-8') as f:
            self._instructions = json.load(f)

        self._confirm_well_formed(json_file, self._instructions)
        if not self._instructions or not all(self._instructions.values()):
            raise ValueError(
                f'Instruction bank {json_file} is empty or malformed.')

    def sample(self, task_type: str) -> Instruction:
        """Sample an instruction (and its language) from the bank.

        Args:
            task_type: The name of the JSON field with relevant instructions.

        Returns:
            A 2-tuple: instruction text and its two-letter language code.
        """
        universe = _get_all_lang_instructions(
            task_type,
            self._instructions,
            self._instruction_bank_size,
        )
        instruction, lang = self._random.choice(universe)
        return Instruction(instruction=instruction, lang=lang)

    def _confirm_well_formed(
            self,
            json_file: str,
            instructions: dict[str, dict[str, list[str]]],
    ) -> None:
        # Check that there are any languages.
        if not instructions:
            raise ValueError(f'Instruction bank {json_file} is empty.')
        # Check that all languages have at least one instruction group.
        if not all(instructions.values()):
            empty = {
                lang
                for lang, groups in instructions.items()
                if not groups
            }
            raise ValueError(
                f'No instructions for language(s) {empty} in {json_file}.')
        # Check that all instruction groups are non-empty.
        for lang, groups in instructions.items():
            if not all(groups.values()):
                empty = {
                    group
                    for group, options in groups.items()
                    if not options
                }
                raise ValueError(
                    f'No instructions for group(s) {empty}'
                    f' in language {lang} in {json_file}')
        # Check that all instructions are non-empty and warn if they might be bad f-strings.
        for lang, groups in instructions.items():
            for group, options in groups.items():
                if not all(options):
                    raise ValueError(
                        f'Found empty instruction for group {group} in language'
                        f' {lang} in {json_file}')
                if any('{' in option for option in options):
                    for option in options:
                        if '{' in option:
                            logging.warning('Open-brace present in instruction. '
                                            'Was an f-string moved into the JSON incorrectly? See: %s', option)
