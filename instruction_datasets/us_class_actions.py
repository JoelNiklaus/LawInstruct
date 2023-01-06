from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class USClassActions(AbstractDataset):
    # Legal Judgement Prediction: US Class Actions

    def __init__(self):
        super().__init__("USClassActions", "https://huggingface.co/datasets/darrow-ai/USClassActions")

    def get_data(self):
        df = load_dataset("darrow-ai/USClassActions", split="train")
        task_type = TASK_TYPE.TEXT_CLASSIFICATION
        jurisdiction = JURISDICTION.US
        instruction_bank = [
            "Read the following United States class action complaint. Predict whether the complaint will be won or not. Output \"win\" or \"lose\".",
            "Will this class action complaint be successful in U.S. Court?"]
        for example in df:
            text = f"{self.random.choice(instruction_bank)}\n\n{example['target_text']}\n\nLikely Verdict: {example['verdict']}"
            prompt_language = "en"
            yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)
