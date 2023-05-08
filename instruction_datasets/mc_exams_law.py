import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class MCExamsLaw(AbstractDataset):

    def __init__(self):
        # TODO do we have an url here: hand built by peter
        super().__init__("MCExamsLaw", "mc_exams_law")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(
            f"{self.raw_data_dir}/raw_legal_mc_with_explanations.csv")
        task_type = TaskType.MULTIPLE_CHOICE
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        for idx, row in df.iterrows():
            q, a, explanation, source = row["Question"], row["Answer"], row[
                "Explanation"], row["Source"]

            # No chain of thought
            instruction, instruction_language = instructions.sample('mc_exams_law_noexplain')
            prompt = f"Q: {q}"
            answer = f"A: {a}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction)

            # Chain of thought
            instruction_expl, instruction_language = instructions.sample('mc_exams_law_explain')
            prompt = f"Q: {q}"
            answer = f"Explanation: {explanation}\nA:{a}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction_expl, prompt, answer,
                                        task_type, jurisdiction)
