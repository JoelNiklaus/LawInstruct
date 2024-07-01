from datasets import load_dataset

from abstract_dataset import AbstractDataset, DataPoint
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

class LegalLensNLI(AbstractDataset): 

    def __init__(self):
        super().__init__(
            "LegalLensNLI",
            "https://huggingface.co/datasets/darrow-ai/LegalLensNLI"
        )

    def get_data(self, instructions: instruction_manager.InstructionManager): 
        task_type = TaskType.NATURAL_LANGUAGE_INFERENCE
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"
        answer_language = "en"

        ds = load_dataset("darrow-ai/LegalLensNLI", split = 'train')
        for example in ds:
            subset = 'legal_lens_nli'
            instruction, instruction_language = instruction.sample(subset)
            promt = f"Case Context: {example['premise']}\n\n" \
                    f"Hypothesis: {example['hypothesis']}\n\n"\
                    f"Relevant Statute: {example['legal_act']}"
            answer = f"Classification: {example['label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        promt, answer, task_type, 
                                        jurisdiction, subset)