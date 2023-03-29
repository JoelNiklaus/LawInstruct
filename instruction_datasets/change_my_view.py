import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class ChangeMyView(AbstractDataset):

    def __init__(self):
        super().__init__("ChangeMyView",
                         "https://chenhaot.com/pages/changemyview.html")

    def get_data(self):
        # ChangeMyView Argumentation
        print("############################")
        print("########## ChangeMyView ###########")
        print("############################")
        instruction_bank = [
            "You are given a position, create an argument that would change the original poster's mind.",
            "Write a counter argument to the proposal.",
            "Write a counter argument to the r/changemyview post.",
            "Write a counterargument to this reddit post."
        ]
        task_type = TaskType.ARGUMENTATION
        jurisdiction = Jurisdiction.UNKNOWN
        instruction_language = "en"
        prompt_language = "en"

        with open(f"{self.raw_data_dir}/train_pair_data.jsonlist") as f:
            x = [json.loads(s) for s in f.readlines()]
            for d in x:
                if isinstance(d['positive']['comments'][0]['body'], list):
                    body = d['positive']['comments'][0]['body'][0].strip()
                else:
                    body = d['positive']['comments'][0]['body'].strip()
                op = d['op_text'].split("EDIT:")[0].strip()
                instruction = self.random.choice(instruction_bank)
                prompt = f"Argument: {op}"
                answer = f"Counter-argument: {body}"
                yield self.build_data_point(instruction_language, prompt_language, "en", instruction,
                                            prompt, answer, task_type, jurisdiction)
