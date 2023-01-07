from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class GSM8K(AbstractDataset):
    def __init__(self):
        super().__init__("GSM8K", "https://huggingface.co/datasets/gsm8k")

    def get_data(self):
        # Add math-type reasoning b/c tax has that flavor
        x = load_dataset("gsm8k", "main", split="train")
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.N_A
        prompt_language = "en"

        instruction_bank = ["Answer the question, make sure to show your work.",
                            "Answer the math question step by step. Show your work.",
                            "Answer the following question in logical steps.",
                            "Answer the following questions."]
        for example in x:
            text = f"{self.random.choice(instruction_bank)}\n\nQ: {example['question']}\nA: {example['answer']}"
            yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)

        x = load_dataset("gsm8k", "socratic", split="train")

        instruction_bank = ["Answer the question, make sure to ask yourself follow up questions.",
                            "Answer the math question using the socratic method. Show your work.",
                            "Answer the following question in logical steps.",
                            "Answer the following questions. Make sure to ask any follow up questions as needed."]
        for example in x:
            text = f"{self.random.choice(instruction_bank)}\n\nQ: {example['question']}\nA: {example['answer']}"
            yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)
