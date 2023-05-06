from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class RedditLegalQA(AbstractDataset):

    def __init__(self):
        super().__init__(
            "RedditLegalQA",
            "https://huggingface.co/datasets/pile-of-law/pile-of-law")

    def get_data(self, instructions: instruction_manager.InstructionManager):

        df = load_dataset("pile-of-law/pile-of-law",
                          "r_legaladvice",
                          split="train")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.UNKNOWN
        instruction_language: str
        instruction: str
        prompt_language = "en"

        for example in df["text"]:
            question = example.split("Question:")[-1]
            q = question.split("Answer #")[0]
            if "deleted" in example.lower() or "removed" in example.lower():
                continue
            answers = question.split("Answer #")[1:]
            answers = [a.split(":")[-1] for a in answers]
            for a in answers:
                instruction, instruction_language = instructions.sample("reddit_legal_qa")
                text = f"Question: {q}\n\nAnalysis: {a}"
                prompt = f"Question: {q}"
                answer = f"Analysis: {a}"
                yield self.build_data_point(instruction_language,
                                            prompt_language,
                                            "en",
                                            instruction,
                                            prompt,
                                            answer,
                                            task_type,
                                            jurisdiction,
                                            subset="r_legaladvice")
