import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class SpanishLaborLaw(AbstractDataset):

    def __init__(self):
        super().__init__("SpanishLaborLaw",
                         "https://zenodo.org/record/4256718#.Y5PoC7LMIlg")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(f"{self.raw_data_dir}/spanish_legal_qa.csv")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.SPAIN
        instruction: str
        instruction_language: str
        prompt_language = "en"

        for idx, row in df.iterrows():
            question, context, answer = row["Question"], row["context"], row[
                "Answer text"]
            subset = "spanish_labor_law"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Context: {context}\nQ: {question}"
            answer = f"A: {answer}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "es", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
