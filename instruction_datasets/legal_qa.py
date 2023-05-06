import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class LegalQA(AbstractDataset):

    def __init__(self):
        super().__init__("LegalQA", "https://github.com/siatnlp/LegalQA")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(f"{self.raw_data_dir}/LegalQA-all-train.csv")

        df = df[df['label'] == 1]

        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.CHINA
        instruction_language: str
        prompt_language = "en"

        for q, a in zip(df['question: body'], df['answer']):
            instruction, instruction_language = instructions.sample("legal_qa")
            prompt = f"Q: {q}"
            answer = f"A: {a}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "zh", instruction, prompt, answer,
                                        task_type, jurisdiction)
