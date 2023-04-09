import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


def _separate_text_into_pieces(
    example: dict,
    *,
    num_answer_lines: int = 1,
) -> tuple[str, str, str]:
    """Turns a COLIEE example into an instruction, prompt, and answer."""
    text = example['text']
    # The first line of the text is the instruction.
    # The last line of the text is the answer.
    # The part in between is the prompt.
    instruction, rest = text.split("\n", maxsplit=1)
    prompt, *answer_lines = rest.rsplit("\n", maxsplit=num_answer_lines)
    # For question answering, we have to provide both the reasoning and the
    # answer, which are given on separate lines.
    answer = '\n'.join(answer_lines)
    return instruction, prompt, answer


class COLIEE(AbstractDataset):

    def __init__(self):
        super().__init__("COLIEE",
                         "https://sites.ualberta.ca/~rabelo/COLIEE2022/")

    def get_data(self):
        jurisdiction = Jurisdiction.JAPAN
        instruction_language = 'en'
        answer_languages = ["en", "jp"]

        # Given two passages, determine entailment
        task_type = TaskType.NATURAL_LANGUAGE_INFERENCE

        for language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task3/passage_entailment/train_{language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    instruction, prompt, answer = _separate_text_into_pieces(
                        example)
                    # Instruction is EN, passage is EN or JP, answer is EN.
                    yield self.build_data_point(instruction_language, language,
                                                'en', instruction, prompt,
                                                answer, task_type, jurisdiction)

        # Given a legal passage, generate an entailed question
        task_type = TaskType.QUESTION_GENERATION
        for language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task3/generate_entailed_question/train_{language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    instruction, prompt, answer = _separate_text_into_pieces(
                        example)
                    # Instruction is EN; passage and generated question EN or JP
                    yield self.build_data_point(instruction_language, language,
                                                language, instruction, prompt,
                                                answer, task_type, jurisdiction)

        # Given a question, provide the relevant legal rule for answering the question and the answer
        task_type = TaskType.QUESTION_ANSWERING
        # TODO:
        for language in answer_languages:
            with open(f"{self.raw_data_dir}/coliee/task4/train_{language}.jsonl"
                     ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    # One line is the legal reasoning; one is the answer.
                    instruction, prompt, answer = _separate_text_into_pieces(
                        example, num_answer_lines=2)
                    # Difficulty: the legal reasoning can be in EN or JP, but
                    # the yes/no answer is always given in English. Chose to
                    # code this based on the legal reasoning answer.
                    yield self.build_data_point(instruction_language, language,
                                                language, instruction, prompt,
                                                answer, task_type, jurisdiction)
