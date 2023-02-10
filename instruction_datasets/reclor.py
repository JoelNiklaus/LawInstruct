import json

from abstract_dataset import AbstractDataset
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE


class ReClor(AbstractDataset):

    def __init__(self):
        super().__init__("ReClor", "https://github.com/yuweihao/reclor")

    def get_data(self):
        ### Reclor has logical reasoning.
        instruction_bank = [
            "Given the context answer the reasoning question.",
            "Answer the logical reasoning multiple choice questions.",
            "State the answer in the following format, \"Final Answer: The final answer is ([ANSWER]). I hope it is correct.\"",
            "Read the passage any any relevant rules describing the world. Apply the rules to the facts to answer the question."
        ]
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.N_A
        prompt_language = "en"

        with open(f"{self.raw_data_dir}/reclor_train.json", "r") as f:
            df = json.loads(f.read())
        for data in df:
            options = ""
            options_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
            for x, lab in zip(data["answers"], options_labels):
                options += f"{lab} {x}\n"
            correct_option = options_labels[data['label']]
            text = f"{self.random.choice(instruction_bank)}\n\nQuestion: {data['context']} {data['question']}\n{options}\nFinal Answer: The final answer is: {correct_option}. I hope it is correct."
            yield self.build_data_point(prompt_language, "en", text, task_type,
                                        jurisdiction)
