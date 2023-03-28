import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


_BLANK_INSTRUCTION = ''


class COLIEE(AbstractDataset):

    def __init__(self):
        super().__init__("COLIEE",
                         "https://sites.ualberta.ca/~rabelo/COLIEE2022/")

    def get_data(self):
        jurisdiction = Jurisdiction.JAPAN
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
                    text = example['text']
                    yield self.build_data_point(prompt_language,
                                                answer_language,
                                                _BLANK_INSTRUCTION, text,
                                                task_type, jurisdiction)

        # Given a legal passage, generate an entailed question
        task_type = TaskType.QUESTION_GENERATION
        for answer_language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task3/generate_entailed_question/train_{answer_language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    text = example['text']
                    yield self.build_data_point(prompt_language,
                                                answer_language,
                                                _BLANK_INSTRUCTION, text,
                                                task_type, jurisdiction)

        # Given a question, provide the relevant legal rule for answering the question and the answer
        task_type = TaskType.QUESTION_ANSWERING
        for answer_language in answer_languages:
            with open(
                    f"{self.raw_data_dir}/coliee/task4/train_{answer_language}.jsonl"
            ) as f:
                examples = [json.loads(x) for x in f.readlines()]
                for example in examples:
                    text = example['text']
                    yield self.build_data_point(prompt_language,
                                                answer_language,
                                                _BLANK_INSTRUCTION, text,
                                                task_type, jurisdiction)
