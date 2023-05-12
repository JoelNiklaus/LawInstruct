import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class KoreanLegalQA(AbstractDataset):

    def __init__(self):
        super().__init__(
            "KoreanLegalQA",
            "https://raw.githubusercontent.com/haven-jeon/LegalQA/main/data/legalqa.jsonlines"
        )

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.SOUTH_KOREA
        instruction_language: str
        prompt_language = "en"

        with open(f"{self.raw_data_dir}/legalqa.jsonlines", "r") as f:
            questions = [json.loads(x) for x in f.readlines()]

        for question in questions:
            subset = "korean_legal_qa"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Q: {question['question']}"
            answer = f"A: {question['answer']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "ko", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
