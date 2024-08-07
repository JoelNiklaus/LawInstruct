from typing import Final

from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

INSTRUCTION_GROUPS: Final[tuple[str, ...]] = (
    'ecthr_a', 'ecthr_b', 'scotus', 'eurlex', 'ledgar', 'unfair_tos', 'case_hold'
)

TASK_CODE_MAPPING = {
    'ecthr_a': 'MLTC',
    'ecthr_b': 'MLTC',
    'scotus': 'SLTC',
    'eurlex': 'MLTC',
    'ledgar': 'SLTC',
    'unfair_tos': 'MLTC',
    'case_hold': 'QA',
}

JURISDICTION_MAPPING = {
    'ecthr_a': Jurisdiction.EU,
    'ecthr_b': Jurisdiction.EU,
    'scotus': Jurisdiction.US,
    'eurlex': Jurisdiction.EU,
    'ledgar': Jurisdiction.US,
    'unfair_tos': Jurisdiction.UNKNOWN,
    'case_hold': Jurisdiction.US,
}


class LexGLUE(AbstractDataset):
    # case_hold is already in natural instructions

    def __init__(self):
        super().__init__("LexGLUE", "https://huggingface.co/datasets/lex_glue")

    def get_data(self, instructions_: instruction_manager.InstructionManager):
        task_type = TaskType.TEXT_CLASSIFICATION
        for subset in INSTRUCTION_GROUPS:
            dataset = load_dataset("lex_glue", subset, split="train")
            task_code = TASK_CODE_MAPPING[subset]
            jurisdiction = JURISDICTION_MAPPING[subset]

            if task_code in ['SLTC', 'QA']:
                class_label = dataset.features["label"]

            for example in dataset:
                # get correct labels
                if task_code in ['SLTC', 'QA']:
                    correct_label = class_label.int2str(
                        example['label'])  # get label name for correct label
                    correct_labels = correct_label if isinstance(
                        correct_label, list) else [correct_label]
                elif task_code == 'MLTC':
                    correct_labels = list(
                        map(str, example['labels']
                            ))  # here we don't have any mapping to label names

                if subset in ['ecthr_a', 'ecthr_b', 'scotus', 'unfair_tos', 'case_hold']:
                    correct_labels = [chr(int(num) + 65) for num in correct_labels]  # convert to letters
                if not correct_labels:
                    correct_labels = ["n/a"]  # set to n/a if no labels

                if task_code in ['SLTC', 'MLTC']:
                    input_text = example["text"]
                    if "ecthr" in subset:
                        input_text = " ".join(input_text)
                elif task_code == 'QA':
                    endings = [f"{chr(i + 65)}: {end}" for i, end in enumerate(example["endings"])]
                    input_text = example["context"] + "\n\nHoldings:\n" + "\n".join(endings)

                prompt = f"Passage: {input_text}"
                answer = f"Label(s): {','.join(correct_labels)}"

                prompt_language = "en"
                instruction, instruction_language = instructions_.sample(subset)
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)
