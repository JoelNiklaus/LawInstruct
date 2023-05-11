import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class CAIL2019(AbstractDataset):

    def __init__(self):
        super().__init__("CAIL2019",
                         "https://github.com/china-ai-law-challenge/CAIL2019")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.CHINA
        instruction_language: str
        answer_language = "zh"
        prompt_language = "zh"

        with open(f"{self.raw_data_dir}/big_train_data.json", "r") as f:
            data = json.loads(f.read())["data"]
            for d in data:
                for paragraph in d['paragraphs']:
                    for question in paragraph['qas']:
                        if question['is_impossible']:
                            answer = "Sorry, this question cannot be answered based on the information available."
                        else:
                            answer = ", ".join(
                                [a['text'] for a in question['answers']])
                        instruction, instruction_language = instructions.sample("cail_2019")
                        prompt = f"{paragraph['context']}\n\nQuestion: {question['question']}"
                        answer = f"Answer: {answer}"
                        yield self.build_data_point(instruction_language,
                                                    prompt_language, answer_language,
                                                    instruction, prompt, answer,
                                                    task_type, jurisdiction)
