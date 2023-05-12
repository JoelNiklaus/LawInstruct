from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class ContractNLI(AbstractDataset):

    def __init__(self):
        super().__init__(
            "ContractNLI",
            "https://huggingface.co/datasets/kiddothe2b/contract-nli")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.NATURAL_LANGUAGE_INFERENCE
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        for subset in ["contractnli_a", "contractnli_b"]:
            df = load_dataset("kiddothe2b/contract-nli", subset, split="train")
            class_label = df.features["label"]
            for example in df:
                subset = 'contract_nli'
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Contract Passage: {example['premise']}\n\n" \
                       f"Hypothesis: {example['hypothesis']}"
                answer = f"Entailment: {class_label.int2str(example['label'])}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)
