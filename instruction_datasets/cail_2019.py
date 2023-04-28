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
        instruction_language = "en"
        prompt_language = "en"

        instruction_bank = [
            "Consider the following passage from a Chinese legal case. Answer the questions about the case. If you cannot answer the question feel free to say as such.",
            "Consider the following situation in Chinese law, answer the questions. If the information is not in the passage, respond with, \"Sorry, this question cannot be answered based on the information available.\"",
            "Consider the following passage from a Chinese legal case. Answer the questions about the case. If the question is impossible to answer, say that it cannot be answered."
        ]
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
                        instruction = self.random.choice(instruction_bank)
                        prompt = f"{paragraph['context']}\n\nQuestion: {question['question']}"
                        answer = f"Answer: {answer}"
                        yield self.build_data_point(instruction_language,
                                                    prompt_language, "zh",
                                                    instruction, prompt, answer,
                                                    task_type, jurisdiction)
