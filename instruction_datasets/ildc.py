import pandas as pd
from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class ILDC(AbstractDataset):
    def __init__(self):
        super().__init__("ILDC", "https://github.com/Exploration-Lab/CJPE")

    def get_data(self):
        df1 = pd.read_csv("raw_data/ILDC_multi.csv")
        df1 = df1[df1["split"] == "train"]
        df2 = pd.read_csv("raw_data/ILDC_single.csv")
        df2 = df2[df2["split"] == "train"]

        task_type = TASK_TYPE.TEXT_CLASSIFICATION
        jurisdiction = JURISDICTION.INDIA
        prompt_language = "en"

        instruction_bank = [
            "According to Indian law, will this petition be accepted? If there is more than one petition consider whether the court will accept at least one.",
            "Will the court accept or reject this petition? Use Indian law. If there is more than one petition consider whether the court will accept at least one."]

        for idx, row in df1.iterrows():
            decision = "Court Decision: Reject" if row["label"] == 0 else "Court Decision: Accept"
            datapoint = f"{self.random.choice(instruction_bank)}\n\n{row['text']}\n\n{decision}"
            yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)

        for idx, row in df2.iterrows():
            decision = "Court Decision: Reject" if row["label"] == 0 else "Court Decision: Accept"
            datapoint = f"{self.random.choice(instruction_bank)}\n\n{row['text']}\n\n{decision}"
            yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)
