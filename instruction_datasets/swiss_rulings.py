from datasets import load_dataset

import instruction_manager
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''

canton_mapping = {
    "AG": "Aargau",
    "AI": "Appenzell Innerrhoden",
    "AR": "Appenzell Ausserrhoden",
    "BE": "Bern",
    "BL": "Basel-Landschaft",
    "BS": "Basel-Stadt",
    "FR": "Fribourg",
    "GE": "Geneva",
    "GL": "Glarus",
    "GR": "Graubünden",
    "JU": "Jura",
    "LU": "Lucerne",
    "NE": "Neuchâtel",
    "NW": "Nidwalden",
    "OW": "Obwalden",
    "SG": "St. Gallen",
    "SH": "Schaffhausen",
    "SO": "Solothurn",
    "SZ": "Schwyz",
    "TG": "Thurgau",
    "TI": "Ticino",
    "UR": "Uri",
    "VD": "Vaud",
    "VS": "Valais",
    "ZG": "Zug",
    "ZH": "Zurich",
    "CH": "Federation",
    "IK": "Intercantonal",
}


def get_canton_name(canton_code):
    return canton_mapping[canton_code.upper()]


class SwissRulings(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissRulings",
            "https://huggingface.co/datasets/rcds/swiss_rulings")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        answer_language = "en"

        # TODO: In the future we could also let it predict the court and the chamber

        df = load_dataset('rcds/swiss_rulings', 'full', split='train')
        for example in df:
            if example['canton'] and example['canton'] != "n/a":
                instruction, instruction_language = instructions.sample("swiss_judgment_location")
                answer = f"Canton: {get_canton_name(example['canton'])}. Region: {example['region']}"

                prompt = f"Facts: {example['facts']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

                prompt = f"Considerations: {example['considerations']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

            if example['topic']:
                instruction, instruction_language = instructions.sample("swiss_judgment_topic")
                answer = f"Topic: {example['topic']}"

                prompt = f"Facts: {example['facts']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)

                prompt = f"Considerations: {example['considerations']}"
                yield self.build_data_point(instruction_language, example["language"], answer_language,
                                            instruction, prompt, answer, task_type, jurisdiction)
