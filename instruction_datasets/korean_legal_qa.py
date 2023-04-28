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
        instruction_language = "en"
        prompt_language = "en"

        instruction_bank = [
            "Consider the following question. Retrieve the relevant South Korean legal article.",
            "What is the best South Korean law that can help answer this question.",
            "What South Korean law best applies."
        ]

        with open(f"{self.raw_data_dir}/legalqa.jsonlines", "r") as f:
            questions = [json.loads(x) for x in f.readlines()]

        for question in questions:
            instruction = self.random.choice(instruction_bank)
            prompt = f"Q: {question['question']}"
            answer = f"A: {question['answer']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "ko", instruction, prompt, answer,
                                        task_type, jurisdiction)
