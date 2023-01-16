from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class ContractNLI(AbstractDataset):
    def __init__(self):
        super().__init__("ContractNLI", "https://huggingface.co/datasets/kiddothe2b/contract-nli")

    def get_data(self):
        task_type = TASK_TYPE.NATURAL_LANGUAGE_INFERENCE
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        for subset in ["contractnli_a", "contractnli_b"]:
            df = load_dataset("kiddothe2b/contract-nli", subset, split="train")
            class_label = df.features["label"]
            instruction_bank = [
                "Consider the following Contract Passage and Hypothesis. Predict whether the Contract Passage entails/contradicts/is neutral to the Hypothesis (entailment, contradiction or neutral).",
                "Does the following Contract Passage entail/contradict/stand neutral to the Hypothesis?"]
            for example in df:
                text = f"{self.random.choice(instruction_bank)}\n\n" \
                       f"Contract Passage: {example['premise']}\n\n" \
                       f"Hypothesis: {example['hypothesis']}\n\n" \
                       f"Entailment: {class_label.int2str(example['label'])}"
                yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)