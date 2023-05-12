from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class GSM8K(AbstractDataset):

    def __init__(self):
        super().__init__("GSM8K", "https://huggingface.co/datasets/gsm8k")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # Add math-type reasoning b/c tax has that flavor
        x = load_dataset("gsm8k", "main", split="train")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.N_A
        instruction_language: str
        prompt_language = "en"

        for example in x:
            subset = "gsm8k_1"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Q: {example['question']}"
            answer = f"A: {example['answer']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)

        x = load_dataset("gsm8k", "socratic", split="train")

        for example in x:
            subset = "gsm8k_2"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Q: {example['question']}"
            answer = f"A: {example['answer']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
