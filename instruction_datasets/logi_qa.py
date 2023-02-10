from abstract_dataset import AbstractDataset
from abstract_dataset import JURISDICTION
from abstract_dataset import TASK_TYPE


class LogiQA(AbstractDataset):

    def __init__(self):
        super().__init__("LogiQA", "https://github.com/lgw863/LogiQA-dataset")

    def get_data(self):
        # Chinese Bar Exam, no explanations.
        instruction_bank = [
            "Answer these multiple choice reasoning questions about Chinese Law. There is only one right answer.",
            "Answer these Chinese Law multiple choice questions. There is only one correct answer. Denote your answer as \"Answer: [answer].\""
        ]
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.CHINA
        prompt_language = "en"

        with open(f"{self.raw_data_dir}/zh_train.txt", "r") as f:
            x = f.readlines()
            i = 0
            while True:
                blank = x[i]
                i += 1
                correct = x[i]
                i += 1
                context = x[i]
                i += 1
                question = x[i]
                i += 1
                choices = []
                for z in range(4):
                    choices.append(x[i])
                    i += 1
                text = f"{self.random.choice(instruction_bank)}\n\nQuestion: {context.strip()} {question}{''.join(choices)}\n\nAnswer: ({correct.strip()})."
                yield self.build_data_point(prompt_language, "zh", text,
                                            task_type, jurisdiction)
                if i >= len(x): break
