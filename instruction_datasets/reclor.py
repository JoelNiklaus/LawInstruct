import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class ReClor(AbstractDataset):

    def __init__(self):
        super().__init__("ReClor", "https://github.com/yuweihao/reclor")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        ### Reclor has logical reasoning.
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.N_A
        instruction_language: str
        prompt_language = "en"

        with open(f"{self.raw_data_dir}/reclor_train.json", "r") as f:
            df = json.loads(f.read())
        for data in df:
            options = ""
            options_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
            for x, lab in zip(data["answers"], options_labels):
                options += f"{lab} {x}\n"
            correct_option = options_labels[data['label']]
            instruction, instruction_language = instructions.sample('reclor')
            prompt = f"Question: {data['context']} {data['question']}\n{options}"
            answer = f"Final Answer: The final answer is: {correct_option}. I hope it is correct."
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction)
