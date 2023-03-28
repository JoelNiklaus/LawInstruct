from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

instructions_for_subsets = {
    "ecthr_a":
        "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
        "Predict the articles of the ECtHR that were violated (if any).",
    "ecthr_b":
        "In this task, you are given the facts from a case heard at the European Court of Human Rights (ECtHR). "
        "Predict the articles of ECtHR that were allegedly violated (considered by the court).",
    "scotus":
        "In this task, you are given a case heard at the Supreme Court of the United States (SCOTUS). "
        "Predict the relevant issue area.",
    "eurlex":
        "In this task, you are given an EU law document published in the EUR-Lex portal. "
        "Predict the relevant EuroVoc concepts.",
    "ledgar":
        "In this task, you are given a contract provision from contracts obtained from US Securities and Exchange Commission (SEC) filings."
        "Predict the main topic.",
    "unfair_tos":
        "In this task, you are given a sentence from a Terms of Service (ToS) document from on-line platforms. "
        "Predict the types of unfair contractual terms",
}

TASK_CODE_MAPPING = {
    'ecthr_a': 'MLTC',
    'ecthr_b': 'MLTC',
    'scotus': 'SLTC',
    'eurlex': 'MLTC',
    'ledgar': 'SLTC',
    'unfair_tos': 'MLTC',
}

JURISDICTION_MAPPING = {
    'ecthr_a': Jurisdiction.EU,
    'ecthr_b': Jurisdiction.EU,
    'scotus': Jurisdiction.US,
    'eurlex': Jurisdiction.EU,
    'ledgar': Jurisdiction.US,
    'unfair_tos': Jurisdiction.UNKNOWN,
}


class LexGLUE(AbstractDataset):
    # case_hold is already in natural instructions

    def __init__(self):
        super().__init__("LexGLUE", "https://huggingface.co/datasets/lex_glue")

    def get_data(self):
        task_type = TaskType.TEXT_CLASSIFICATION
        for subset, instructions in instructions_for_subsets.items():
            dataset = load_dataset("lex_glue", subset, split="train")
            task_code = TASK_CODE_MAPPING[subset]
            jurisdiction = JURISDICTION_MAPPING[subset]

            if task_code == 'SLTC':
                class_label = dataset.features["label"]

            for example in dataset:
                # get correct labels
                if task_code == 'SLTC':
                    correct_label = class_label.int2str(
                        example['label'])  # get label name for correct label
                    correct_labels = correct_label if isinstance(
                        correct_label, list) else [correct_label]
                elif task_code == 'MLTC':
                    correct_labels = list(
                        map(str, example['labels']
                           ))  # here we don't have any mapping to label names

                input_text = example['text']
                if 'ecthr' in subset:
                    input_text = " ".join(input_text)
                answer = f"Passage {input_text} Labels: {','.join(correct_labels)}"

                text = f"{instructions}\n\n{answer}"
                prompt_language = "en"
                yield self.build_data_point(prompt_language, "en", text,
                                            task_type, jurisdiction, subset)
