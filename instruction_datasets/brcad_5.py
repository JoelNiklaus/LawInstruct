from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager
import multiple_choice

_MC_OPTIONS = ['NÃO PROVIMENTO', 'PROVIMENTO']


class BrCAD5(AbstractDataset):

    def __init__(self):
        super().__init__("BrCAD5",
                         "https://huggingface.co/datasets/joelito/BrCAD-5")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('joelito/BrCAD-5', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.BRAZIL
        instruction_language: str
        prompt_language = "pt"
        answer_language = "pt"

        for example in df:
            case = example['preprocessed_full_text_first_instance_court_ruling']

            subset = "brcad5_judgment"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Case: {case}"
            answer = f"Judgement: {example['label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

        for example in df:
            case = example['preprocessed_full_text_first_instance_court_ruling']
            subset = "brcad5_law_area"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Case: {case}"
            answer = f"Area of Law: {example['current_case_class']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

        for example in df:
            case = example['preprocessed_full_text_first_instance_court_ruling']
            for level in ["1st", "2nd", "3rd"]:
                instruction, instruction_language = instructions.sample(f"brcad5_topic_{level}")
                prompt = f"Case: {case}"
                answer = f"Topic: {example[f'case_topic_{level}_level']}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, answer_language,
                                            instruction, prompt, answer,
                                            task_type, jurisdiction, "brcad5_topic")

        for example in df:
            case = example['preprocessed_full_text_first_instance_court_ruling']
            markers = multiple_choice.sample_markers_for_options(_MC_OPTIONS)
            outcome_mc1 = markers[['NÃO PROVIMENTO',
                                          'PROVIMENTO'].index(example["label"])]
            subset = "brcad5_mc"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {case} How would the court find?\n{markers[0]} The court should dismiss the case.\n{markers[1]} The court should affirm the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

            outcome_mc1 = list(reversed(markers))[['NÃO PROVIMENTO',
                                          'PROVIMENTO'].index(example["label"])]
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {case} How would the court find?\n{markers[0]} The court should approve the case.\n{markers[1]} The court should dismiss the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)
