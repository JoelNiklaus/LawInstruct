import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


def _separate_text_into_pieces(example: dict) -> tuple[str, str, str]:
    """Turns a COLIEE example into an instruction, prompt, and answer."""
    text = example['text']
    # The first line of the text is the instruction.
    # The last line of the text is the answer.
    # The part in between is the prompt.
    instruction, rest = text.split("\n", maxsplit=1)
    prompt, answer = rest.rsplit("\n", maxsplit=1)
    return instruction, prompt, answer


class COLIEE(AbstractDataset):

    def __init__(self):
        super().__init__("COLIEE",
                         "https://sites.ualberta.ca/~rabelo/COLIEE2022/")

    def get_data(self):
        jurisdiction = Jurisdiction.JAPAN
        instruction_language = 'en'
        prompt_language = "en"

        answer_languages = ["en", "jp"]

        # Given two passages, determine entailment
        task_type = TaskType.NATURAL_LANGUAGE_INFERENCE

        for answer_language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task3/passage_entailment/train_{answer_language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    instruction, prompt, answer = _separate_text_into_pieces(
                        example)
                    yield self.build_data_point(instruction_language,
                                                prompt_language,
                                                answer_language,
                                                instruction,
                                                prompt, answer, task_type,
                                                jurisdiction)

        # Given a legal passage, generate an entailed question
        task_type = TaskType.QUESTION_GENERATION
        for answer_language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task3/generate_entailed_question/train_{answer_language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    instruction, prompt, answer = _separate_text_into_pieces(example)
                    yield self.build_data_point(instruction_language,
                                                prompt_language,
                                                answer_language,
                                                instruction,
                                                prompt, answer, task_type,
                                                jurisdiction)

        # Given a question, provide the relevant legal rule for answering the question and the answer
        task_type = TaskType.QUESTION_ANSWERING
        for answer_language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task4/train_{answer_language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    instruction, prompt, answer = _separate_text_into_pieces(example)
                    yield self.build_data_point(instruction_language,
                                                prompt_language,
                                                answer_language,
                                                instruction,
                                                prompt, answer, task_type,
                                                jurisdiction)
