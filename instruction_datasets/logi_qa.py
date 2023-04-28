from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class LogiQA(AbstractDataset):

    def __init__(self):
        super().__init__("LogiQA", "https://github.com/lgw863/LogiQA-dataset")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # Chinese Bar Exam, no explanations.
        instruction_bank = [
            "Answer these multiple choice reasoning questions about Chinese Law. There is only one right answer.",
            "Answer these Chinese Law multiple choice questions. There is only one correct answer. Denote your answer as \"Answer: [answer].\""
        ]
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.CHINA
        instruction_language = "en"
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
                instruction = self.random.choice(instruction_bank)
                prompt = f"Question: {context.strip()} {question}{''.join(choices)}"
                answer = f"Answer: ({correct.strip()})."
                yield self.build_data_point(instruction_language,
                                            prompt_language, "zh", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction)
                if i >= len(x):
                    break
