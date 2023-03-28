from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


def get_multiple_choice_instruction_bank():
    return [
        "Please answer these multiple choice questions. Denote the correct answer as \"Answer\".",
        "Pick the most likely correct answer."
    ]


class BrCAD5(AbstractDataset):

    def __init__(self):
        super().__init__("BrCAD5",
                         "https://huggingface.co/datasets/joelito/BrCAD-5")

    def get_data(self):
        df = load_dataset('joelito/BrCAD-5', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.BRAZIL
        prompt_language = "en"
        answer_language = "pt"

        for example in df:
            case = example['preprocessed_full_text_first_instance_court_ruling']
            instruction = 'Determine what you think the Brazilian appeals court will rule for the case.'
            text = f"Case:{case}\nJudgement: {example['label']}"
            yield self.build_data_point(prompt_language, answer_language,
                                        instruction, text, task_type,
                                        jurisdiction)

            # TODO: this isn't _really_ an instruction, just a question...
            instruction = 'What area of law is this case related to?'
            text = f"Case:{case}\nArea of Law: {example['current_case_class']}"
            yield self.build_data_point(prompt_language, answer_language,
                                        instruction, text, task_type,
                                        jurisdiction)

            for level in ["1st", "2nd", "3rd"]:
                # TODO: this isn't _really_ an instruction, just a question...
                instruction = f"What {level}-level topic is this case related to?"
                text = f"Case:{case}\nTopic: {example[f'case_topic_{level}_level']}"
                yield self.build_data_point(prompt_language, answer_language,
                                            instruction, text, task_type,
                                            jurisdiction)

            outcome_mc1 = ["(a)", "(b)"][['NÃO PROVIMENTO',
                                          'PROVIMENTO'].index(example["label"])]
            instruction = self.random.choice(
                get_multiple_choice_instruction_bank())
            text = f"Question: {case} How would the court find?\n(a) The court should dismiss the case.\n(b) The court should affirm the case.\n" \
                   f"Answer: {outcome_mc1}."
            yield self.build_data_point(prompt_language, answer_language,
                                        instruction, text, task_type,
                                        jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][['NÃO PROVIMENTO',
                                          'PROVIMENTO'].index(example["label"])]
            instruction = self.random.choice(
                get_multiple_choice_instruction_bank())
            text = f"Question: {case} How would the court find?\n(a) The court should approve the case.\n(b) The court should dismiss the case.\n" \
                   f"Answer: {outcome_mc1}."
            yield self.build_data_point(prompt_language, answer_language,
                                        instruction, text, task_type,
                                        jurisdiction)
