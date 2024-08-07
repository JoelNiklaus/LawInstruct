from typing import Final

import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

_BLANK_INSTRUCTION = ""


class ValidWills(AbstractDataset):

    def __init__(self):
        super().__init__("ValidWills", "https://arxiv.org/pdf/2210.16989.pdf")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # Will Validity
        train = pd.read_csv(
            f'{self.raw_data_dir}/wills_train.csv',
            encoding='utf-8')  # replace with real path and dataset names
        subset = "valid_wills_entailment"
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.US
        prompt_language = "en"

        for idx, row in train.iterrows():
            statement, conditions, law, classification = row["statement"], row[
                "conditions"], row["law"], row["classification"]
            CLASSIFICATION_MAP = ['refuted', 'supported', 'unrelated']
            classification = CLASSIFICATION_MAP[classification]
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}"
            prompt2 = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\nIs the statement supported by the law and condition?"
            answer = answer2 = f'Answer: {classification}'

            options_mc = ["supported", "refuted", "unrelated"]
            lookup = ["(a)", "(b)", "(c)"]
            self.random.shuffle(options_mc)
            option_mc_string = ""
            correct_option = None
            for choice_letter, option in zip(lookup, options_mc):
                if option == classification:
                    correct_option = choice_letter
                option_mc_string += f"{choice_letter} {option}\n"
            prompt_mc = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\n{option_mc_string}"
            answer_mc = f'Answer: {correct_option}'
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction, subset)
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", _BLANK_INSTRUCTION, prompt2,
                                        answer2, task_type, jurisdiction, subset)
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", _BLANK_INSTRUCTION, prompt_mc,
                                        answer_mc, task_type, jurisdiction, subset)
