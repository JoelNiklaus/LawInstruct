from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class BrCAD5(AbstractDataset):

    def __init__(self):
        super().__init__("BrCAD5",
                         "https://huggingface.co/datasets/joelito/BrCAD-5")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('joelito/BrCAD-5', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.BRAZIL
        instruction_language: str
        prompt_language = "en"
        answer_language = "pt"

        for example in df:
            case = example['preprocessed_full_text_first_instance_court_ruling']

            instruction, instruction_language = instructions.sample("brcad5_judgment")
            prompt = f"Case: {case}"
            answer = f"Judgement: {example['label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)

            instruction, instruction_language = instructions.sample("brcad5_law_area")
            prompt = f"Case: {case}"
            answer = f"Area of Law: {example['current_case_class']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)

            for level in ["1st", "2nd", "3rd"]:
                instruction, instruction_language = instructions.sample(f"brcad5_topic_{level}")
                prompt = f"Case: {case}"
                answer = f"Topic: {example[f'case_topic_{level}_level']}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, answer_language,
                                            instruction, prompt, answer,
                                            task_type, jurisdiction)

            outcome_mc1 = ["(a)", "(b)"][['NÃO PROVIMENTO',
                                          'PROVIMENTO'].index(example["label"])]
            instruction, instruction_language = instructions.sample("brcad5_mc")
            prompt = f"Question: {case} How would the court find?\n(a) The court should dismiss the case.\n(b) The court should affirm the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][['NÃO PROVIMENTO',
                                          'PROVIMENTO'].index(example["label"])]
            instruction, instruction_language = instructions.sample("brcad5_mc")
            prompt = f"Question: {case} How would the court find?\n(a) The court should approve the case.\n(b) The court should dismiss the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)
