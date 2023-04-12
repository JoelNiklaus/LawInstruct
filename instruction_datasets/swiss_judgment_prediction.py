from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType

_BLANK_INSTRUCTION = ''


def get_multiple_choice_instruction_bank():
    return [
        'Please answer these multiple choice questions. Denote the correct answer as "Answer".',
        "Pick the most likely correct answer."
    ]


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
}


def get_canton_name(canton_code):
    return canton_mapping[canton_code.upper()]


class SwissJudgmentPrediction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissJudgmentPrediction",
            "https://huggingface.co/datasets/rcds/swiss_judgment_prediction")

    def get_data(self):
        df = load_dataset('rcds/swiss_judgment_prediction', 'all+mt', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = 'en'
        answer_language = "en"
        for example in df:
            court_location = "" if example['canton'] == "n/a" \
                else f"The lower court is located in {get_canton_name(example['canton'])}."
            judgement = ["dismiss", "approve"][example['label']]
            instruction = f"Determine if you think the Swiss court will dismiss or approve the case. {court_location}"
            prompt = f"Facts: {example['text']}"
            answer = f"Judgement: {judgement}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction,
                                        prompt, answer, task_type, jurisdiction)

            instruction = "What area of law is this case related to?"
            prompt = f"Case: {example['text']}"
            answer = f"Area of Law: {example['legal area']}"
            yield self.build_data_point(instruction_language, example["language"], answer_language,
                                        instruction,
                                        prompt, answer, task_type, jurisdiction)

            if court_location != "":
                instruction = "Where do you think this case was adjudicated?"
                prompt = f"Case: {example['text']}"
                answer = f"Canton: {get_canton_name(example['canton'])}. Region: {example['region']}"
                yield self.build_data_point(instruction_language, example["language"],
                                            answer_language,
                                            instruction,
                                            prompt, answer, task_type,
                                            jurisdiction)

            task_type = TaskType.MULTIPLE_CHOICE
            outcome_mc1 = ["(a)", "(b)"][example["label"]]
            text = example['text']
            instruction = self.random.choice(
                get_multiple_choice_instruction_bank())
            prompt = f"Question: {text} How would the court find?\n" \
                     f"(a) The court should dismiss the case.\n(b) The court should affirm the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, answer_language, example["language"],
                                        instruction,
                                        prompt, answer, task_type, jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][example["label"]]
            text = example['text']
            instruction = self.random.choice(
                get_multiple_choice_instruction_bank())
            prompt = f"Question: {text} How would the court find?\n" \
                     f"(a) The court should approve the case.\n(b) The court should dismiss the case."
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, answer_language, example["language"],
                                        instruction,
                                        prompt, answer, task_type, jurisdiction)
