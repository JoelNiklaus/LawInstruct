from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class RedditLegalQA(AbstractDataset):
    def __init__(self):
        super().__init__("RedditLegalQA", "https://huggingface.co/datasets/pile-of-law/pile-of-law")

    def get_data(self):
        instruction_bank = [
            "Here is someone's legal concern. Answer as if you were replying on Reddit. If you are not a lawyer, include the disclaimer IANAL.",
            "Here is someone's legal question. Advice them on the situation. Think like a lawyer on Reddit."]

        df = load_dataset("pile-of-law/pile-of-law", "r_legaladvice", split="train")
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.UNKNOWN
        prompt_language = "en"

        for example in df["text"]:
            question = example.split("Question:")[-1]
            q = question.split("Answer #")[0]
            if "deleted" in example.lower() or "removed" in example.lower():
                continue
            answers = question.split("Answer #")[1:]
            answers = [a.split(":")[-1] for a in answers]
            for a in answers:
                text = f"{self.random.choice(instruction_bank)}\n\nQuestion: {q}\n\nAnalysis: {a}"
                yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction,
                                            subset="r_legaladvice")