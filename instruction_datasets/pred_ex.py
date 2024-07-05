from collections.abc import Iterator
from datasets import load_dataset

from abstract_dataset import AbstractDataset, DataPoint
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class PredEx(AbstractDataset):

    def __init__(self):
        super().__init__(
            "PredEx", 
            "https://huggingface.co/datasets/L-NLProc/PredEx_Instruction-Tuning_Pred-Exp")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('L-NLProc/PredEx_Instruction-Tuning_Pred-Exp', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.INDIA
        prompt_language = "en"
        answer_language = "en"
        instruction_language: str

        for example in df:
            subset = "predex_judgement"
            case_name = example['Case Name']
            case_text = example['Input']
            case = (case_name or '') + (case_text or '')

            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Case: {case}"
            answer = f"Prediction: {example['Label']}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset) 
        
        task_type = TaskType.ARGUMENTATION
        label_mapping = {0: 'rejected', 1: 'accepted'}
        
        for example in df:
            subset = "predex_explanation"
            case_name = example['Case Name']
            case_text = example['Input']
            case_outcome = label_mapping[example['Label']]
            case = (case_name or '') + (case_text or '')

            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Case: {case}\n\n" \
                    f"Outcome: {case_outcome}"
            answer = f"{example['Output']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset) 
