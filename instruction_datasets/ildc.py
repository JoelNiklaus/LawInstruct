import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class ILDC(AbstractDataset):

    def __init__(self):
        super().__init__("ILDC", "https://github.com/Exploration-Lab/CJPE")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df1 = pd.read_csv(f"{self.raw_data_dir}/ILDC_multi.csv")
        df1 = df1[df1["split"] == "train"]
        df2 = pd.read_csv(f"{self.raw_data_dir}/ILDC_single.csv")
        df2 = df2[df2["split"] == "train"]

        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.INDIA
        instruction_language: str
        prompt_language = "en"

        subset = "ildc"
        for idx, row in df1.iterrows():
            decision = "Court Decision: Reject" if row[
                "label"] == 0 else "Court Decision: Accept"
            instruction, instruction_language = instructions.sample(subset)
            prompt = row['text']
            answer = decision
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)

        for idx, row in df2.iterrows():
            decision = "Court Decision: Reject" if row[
                "label"] == 0 else "Court Decision: Accept"
            instruction, instruction_language = instructions.sample(subset)
            prompt = row['text']
            answer = decision
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
