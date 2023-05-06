from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class ShortAnswerFeedback(AbstractDataset):

    def __init__(self):
        super().__init__(
            "ShortAnswerFeedback",
            "https://huggingface.co/datasets/JohnnyBoy00/saf_legal_domain_german"
        )

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset("JohnnyBoy00/saf_legal_domain_german")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.GERMANY
        instruction_language: str
        instruction: str
        prompt_language = "en"

        for example in df["train"]:
            instruction, instruction_language = instructions.sample('short_answer_feedback_openqa')
            prompt = f"Q: {example['question']}"
            answer = f"A: {example['reference_answer']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "de", instruction, prompt, answer,
                                        task_type, jurisdiction)

            instruction, instruction_language = instructions.sample('short_answer_feedback_rating')

            prompt = f"Q: {example['question']}\nA: {example['provided_answer']}"
            answer = f"Feedback: {example['verification_feedback']}\nScore: {example['score']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "de", instruction, prompt, answer,
                                        task_type, jurisdiction)

            instruction, instruction_language = instructions.sample('short_answer_feedback_error_class')
            prompt = f"Q: {example['question']}\nA: {example['provided_answer']}"
            answer = f"Feedback: {example['verification_feedback']}\nScore: {example['score']}\nError Type: {example['error_class']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "de", instruction, prompt, answer,
                                        task_type, jurisdiction)
