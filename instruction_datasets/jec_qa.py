import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class JECQA(AbstractDataset):

    def __init__(self):
        super().__init__("JECQA", "https://jecqa.thunlp.org/")

    def get_data(self):

        instruction_bank = [
            "Answer these multiple choice reasoning questions about Chinese Law. Select all answers that apply, you may have multiple correct answers.",
            "Answer these Chinese Law multiple choice questions, you might have multiple correct answers. Denote your answer(s) as \"Answer: [answer(s)].\""
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.CHINA
        prompt_language = "en"

        with open(f"{self.raw_data_dir}/jecqa_0_train.json") as f:
            questions = [json.loads(x) for x in f.readlines()]
            with open(f"{self.raw_data_dir}/jecqa_1_train.json") as f:
                questions.extend([json.loads(x) for x in f.readlines()])

        for q in questions:
            prompt = f"{self.random.choice(instruction_bank)}\n\n{q['statement']}\n\n"
            for k, v in q["option_list"].items():
                prompt += f"{k}. {v}\n"
            prompt += "\n\nFinal Answer(s): {','.join(q['answer'])}"
            yield self.build_data_point(prompt_language, "zh", prompt,
                                        task_type, jurisdiction)
