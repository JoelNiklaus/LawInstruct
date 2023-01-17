import json

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class CAIL2019(AbstractDataset):
    def __init__(self):
        super().__init__("CAIL2019", "https://github.com/china-ai-law-challenge/CAIL2019")

    def get_data(self):
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.CHINA
        prompt_language = "en"

        instruction_bank = [
            "Consider the following passage from a Chinese legal case. Answer the questions about the case. If you cannot answer the question feel free to say as such.",
            "Consider the following situation in Chinese law, answer the questions. If the information is not in the passage, respond with, \"Sorry, this question cannot be answered based on the information available.\"",
            "Consider the following passage from a Chinese legal case. Answer the questions about the case. If the question is impossible to answer, say that it cannot be answered."]
        with open("./raw_data/big_train_data.json", "r") as f:
            data = json.loads(f.read())["data"]
            for d in data:
                for paragraph in d['paragraphs']:
                    for question in paragraph['qas']:
                        if question['is_impossible']:
                            answer = "Sorry, this question cannot be answered based on the information available."
                        else:
                            answer = ", ".join([a['text'] for a in question['answers']])
                        text = f"{self.random.choice(instruction_bank)}\n\n{paragraph['context']}\n\nQuestion:{question['question']}\nAnswer:{answer}"
                        yield self.build_data_point(prompt_language, "zh", text, task_type, jurisdiction)
