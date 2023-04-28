from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class LboxOpen(AbstractDataset):

    def __init__(self):
        super().__init__("LboxOpen", "https://github.com/lbox-kr/lbox-open")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # statutes classification task
        data_st_plus = load_dataset("lbox/lbox_open",
                                    "statute_classification_plus")
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SOUTH_KOREA
        instruction_language = "en"
        prompt_language = "en"
        answer_language = "ko"
        instruction_bank = [
            "For the given case facts, predict the related South Korean legal statute.",
            "When presented with this fact pattern, what are the relevant legal statutes in South Korean law?"
        ]

        for x in data_st_plus["train"]:
            instruction = self.random.choice(instruction_bank)
            prompt = f"Fact: {x['facts']}"
            answer = f"Statute(s): {','.join(x['statutes'])}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)

        # Legal judgement prediction tasks
        data_ljp_criminal = load_dataset("lbox/lbox_open", "ljp_criminal")
        instruction_bank = [
            "Given these facts from a South Korean criminal law case. Predict the court's ruling and the reason for the ruling."
        ]
        for x in data_ljp_criminal["train"]:
            reason = ""
            if x["reason"] != "" and x["reason"] != -1:
                reason = f"Reason: {x['reason']}"
            instruction = self.random.choice(instruction_bank)
            prompt = f"Fact: {x['facts']}\n{reason}"
            answer = f"Ruling: {x['ruling']['text']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)

        data_ljp_civil = load_dataset("lbox/lbox_open", "ljp_civil")
        for x in data_ljp_civil["train"]:
            instruction = self.random.choice(instruction_bank)
            prompt = f"Fact: {x['facts'].strip()}\n\nClaim: {x['gist_of_claim']['text'].strip()}"
            answer = f"Ruling: {x['ruling']['text']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)
