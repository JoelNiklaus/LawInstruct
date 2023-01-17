import json
import os

from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class Lila(AbstractDataset):
    def __init__(self):
        super().__init__("Lila", "https://github.com/allenai/Lila")

    def get_data(self):
        print("############################")
        print("########## Lila ###########")
        print("############################")
        json_files = [pos_json for pos_json in os.listdir("raw_data/all_lila/") if pos_json.endswith('.json')]
        instruction_bank = ["Consider the following question. Write a Python program to solve it.",
                            "Write a Python program to solve the following question, denote it as \"Program:\". Provide the output as \"Answer:\"."]
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.N_A
        prompt_language = "en"

        for json_file in json_files:
            with open(os.path.join("raw_data/all_lila/", json_file), "r") as f:
                loaded_file = json.loads(f.read())
                for example in loaded_file["Instances"]:
                    if example["split"] != "train":
                        continue
                    for program, answer in zip(example['Output Program'], example['Output Answer']):
                        datapoint = f"{self.random.choice(instruction_bank)}\n\n" \
                                    f"Question: {example['Input']}\n" \
                                    f"Program:\n```python\n{program}\n```\n" \
                                    f"Answer: {answer}"
                        yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)
