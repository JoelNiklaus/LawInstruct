import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class JECQA(AbstractDataset):

    def __init__(self):
        super().__init__("JECQA", "https://jecqa.thunlp.org/")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.CHINA
        instruction_language: str
        prompt_language = "zh"

        with open(f"{self.raw_data_dir}/jecqa_0_train.json") as f:
            questions = [json.loads(x) for x in f.readlines()]
            with open(f"{self.raw_data_dir}/jecqa_1_train.json") as f:
                questions.extend([json.loads(x) for x in f.readlines()])

        for q in questions:
            instruction, instruction_language = instructions.sample('jec_qa')
            prompt = f"{q['statement']}\n\n"
            for k, v in q["option_list"].items():
                prompt += f"{k}. {v}\n"
            prompt.rstrip("\n")  # remove trailing newline
            answer = f"Final Answer(s): {','.join(q['answer'])}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "zh", instruction, prompt, answer,
                                        task_type, jurisdiction)
