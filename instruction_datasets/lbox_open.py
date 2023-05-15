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
        instruction_language: str
        prompt_language = "ko"
        answer_language = "ko"

        for x in data_st_plus["train"]:
            subset = "lbox_open_statute"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Fact: {x['facts']}"
            answer = f"Statute(s): {','.join(x['statutes'])}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

        # Legal judgement prediction tasks
        data_ljp_criminal = load_dataset("lbox/lbox_open", "ljp_criminal")
        subset = "lbox_open_judgment"
        for x in data_ljp_criminal["train"]:
            reason = ""
            if x["reason"] != "" and x["reason"] != -1:
                reason = f"Reason: {x['reason']}"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Fact: {x['facts']}\n{reason}"
            answer = f"Ruling: {x['ruling']['text']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

        data_ljp_civil = load_dataset("lbox/lbox_open", "ljp_civil")
        for x in data_ljp_civil["train"]:
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Fact: {x['facts'].strip()}\n\nClaim: {x['gist_of_claim']['text'].strip()}"
            answer = f"Ruling: {x['ruling']['text']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)
