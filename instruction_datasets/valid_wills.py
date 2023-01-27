import pandas as pd

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class ValidWills(AbstractDataset):
    def __init__(self):
        super().__init__("ValidWills", "https://arxiv.org/pdf/2210.16989.pdf")

    def get_data(self):
        # Will Validity
        train = pd.read_csv('./raw_data/wills_train.csv', encoding='utf-8')  # replace with real path and dataset names
        instruction_bank = [
            "Given a statement in a will, the relevant U.S. law, is the condition supported, refuted, or unrelated.",
            "Is the statement in the will valid given the law and conditions? Answer with one of unrelated, supported, refuted."]
        task_type = TASK_TYPE.TEXT_CLASSIFICATION
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        for idx, row in train.iterrows():
            statement, conditions, law, classification = row["statement"], row["conditions"], row["law"], row[
                "classification"]
            CLASSIFICATION_MAP = ['refuted', 'supported', 'unrelated']
            classification = CLASSIFICATION_MAP[classification]
            prompt = f"{self.random.choice(instruction_bank)}\n\nStatement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\nAnswer: {classification}"
            prompt2 = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\nIs the statement supported by the law and condition?\n\nAnswer: {classification}"

            options_mc = ["supported", "refuted", "unrelated"]
            lookup = ["(a)", "(b)", "(c)"]
            self.random.shuffle(options_mc)
            option_mc_string = ""
            correct_option = None
            for choice_letter, option in zip(lookup, options_mc):
                if option == classification:
                    correct_option = choice_letter
                option_mc_string += f"{choice_letter} {option}\n"
            prompt_mc = f"Statement: {statement}\n\nLaw: {law}\n\nCondition: {conditions}\n\n{option_mc_string}\n\nAnswer: {correct_option}"
            yield self.build_data_point(prompt_language, "en", prompt, task_type, jurisdiction)
            yield self.build_data_point(prompt_language, "en", prompt2, task_type, jurisdiction)
            yield self.build_data_point(prompt_language, "en", prompt_mc, task_type, jurisdiction)