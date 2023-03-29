import json

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class CAIL2022(AbstractDataset):

    def __init__(self):
        super().__init__(
            "CAIL2022",
            "https://github.com/china-ai-law-challenge/CAIL2022/tree/main/lblj/data/stage_2"
        )

    def get_data(self):
        jurisdiction = Jurisdiction.CHINA
        instruction_language = "en"
        prompt_language = "en"
        answer_language = "zh"

        with open(f"{self.raw_data_dir}/cail2022_train_entry_lblj.jsonl",
                  "r",
                  encoding="utf8") as f:
            questions = [json.loads(x) for x in f.readlines()]

        instruction_bank_mc = [
            "Use Chinese law to answer these multiple choice questions. Pick the best counter-argument to the plaintiff's argument.",
            "Which of these is the best response to the following argument if you were the defendant? Consider Chinese law."
        ]
        instruction_bank = [
            "Use Chinese law. What is the counter-argument to the plaintiff's argument?",
            "How should Defendant respond to the following argument? Use Chinese law."
        ]
        instruction_bank_crime = [
            "Consider Chinese law, what is the likely crime being discussed here."
        ]
        lookup = ["(a)", "(b)", "(c)", "(d)", "(e)"]
        for question in questions:
            task_type = TaskType.MULTIPLE_CHOICE
            instruction = self.random.choice(instruction_bank_mc)
            prompt = f"Plaintiff's Argument:{question['sc']}\n\n(a) {question['bc_1']}\n(b) {question['bc_2']}\n(c) {question['bc_3']}\n(d) {question['bc_4']}\n(e) {question['bc_5']}"
            answer = f"Best counter-argument: {lookup[question['answer'] - 1]}"
            yield self.build_data_point(instruction_language, prompt_language, answer_language,
                                        instruction, prompt, answer, task_type,
                                        jurisdiction)

            task_type = TaskType.QUESTION_ANSWERING
            response = question[f"bc_{question['answer']}"]
            instruction = self.random.choice(instruction_bank)
            prompt = f"Plaintiff's Argument:{question['sc']}"
            answer = f"Defendant's Response: {response}"
            yield self.build_data_point(instruction_language, prompt_language, answer_language,
                                        instruction, prompt, answer, task_type,
                                        jurisdiction)

            task_type = TaskType.TEXT_CLASSIFICATION
            instruction = self.random.choice(instruction_bank_crime)
            prompt = f"Plaintiff's Argument:{question['sc']}\nDefendant's Response: {response}"
            answer = f"Crime: {question['crime']}"
            yield self.build_data_point(instruction_language, prompt_language, answer_language,
                                        instruction, prompt, answer, task_type,
                                        jurisdiction)

            instruction = self.random.choice(instruction_bank_crime)
            prompt = question['sc']
            answer = f"Crime: {question['crime']}"
            yield self.build_data_point(instruction_language, prompt_language, answer_language,
                                        instruction, prompt, answer, task_type,
                                        jurisdiction)
