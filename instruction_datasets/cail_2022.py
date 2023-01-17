import json

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class CAIL2022(AbstractDataset):
    def __init__(self):
        super().__init__("CAIL2022", "https://github.com/china-ai-law-challenge/CAIL2022/tree/main/lblj/data/stage_2")

    def get_data(self):
        jurisdiction = JURISDICTION.CHINA
        prompt_language = "en"
        answer_language = "zh"

        with open("raw_data/cail2022_train_entry_lblj.jsonl", "r", encoding="utf8") as f:
            questions = [json.loads(x) for x in f.readlines()]

        instruction_bank_mc = [
            "Use Chinese law to answer these multiple choice questions. Pick the best counter-argument to the plaintiff's argument.",
            "Which of these is the best response to the following argument if you were the defendant? Consider Chinese law."]
        instruction_bank = ["Use Chinese law. What is the counter-argument to the plaintiff's argument?",
                            "How should Defendant respond to the following argument? Use Chinese law."]
        instruction_bank_crime = ["Consider Chinese law, what is the likely crime being discussed here."]
        lookup = ["(a)", "(b)", "(c)", "(d)", "(e)"]
        for question in questions:
            task_type = TASK_TYPE.MULTIPLE_CHOICE
            datapoint = f"{self.random.choice(instruction_bank_mc)}\n\nPlaintiff's Argument:{question['sc']}\n\n(a) {question['bc_1']}\n(b) {question['bc_2']}\n(c) {question['bc_3']}\n(d) {question['bc_4']}\n(e) {question['bc_5']}"
            datapoint += "Best counter-argument: {lookup[question['answer'] - 1]}"
            yield self.build_data_point(prompt_language, answer_language, datapoint, task_type, jurisdiction)

            task_type = TASK_TYPE.QUESTION_ANSWERING
            response = question[f"bc_{question['answer']}"]
            datapoint = f"{self.random.choice(instruction_bank)}\n\nPlaintiff's Argument:{question['sc']}\nDefendant's Response: {response}"
            yield self.build_data_point(prompt_language, answer_language, datapoint, task_type, jurisdiction)

            task_type = TASK_TYPE.TEXT_CLASSIFICATION
            datapoint = f"{self.random.choice(instruction_bank_crime)}\n\nPlaintiff's Argument:{question['sc']}\nDefendant's Response: {response}\nCrime: {question['crime']}"
            yield self.build_data_point(prompt_language, answer_language, datapoint, task_type, jurisdiction)

            datapoint = f"{self.random.choice(instruction_bank_crime)}\n\n{question['sc']}\nCrime: {question['crime']}"
            yield self.build_data_point(prompt_language, answer_language, datapoint, task_type, jurisdiction)
