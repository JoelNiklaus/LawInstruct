from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

_BLANK_PROMPT = ''


class CaseBriefs(AbstractDataset):

    def __init__(self):
        super().__init__("CaseBriefs", "https://www.oyez.org")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # Case briefs take the form of a question and an answer.
        case_brief_instructions = [
            "Given the key facts of a case, provide the core question the court should answer, then provide an analysis for how the an American court might decide the case.",
            "Given the facts, describe how an American court should think about the key issue?"
        ]

        df = load_dataset("lawinstruct/case-briefs",
                          "combined",
                          use_auth_token=True)
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.US
        instruction_language = 'en'
        prompt_language = "en"

        for example in df["train"]["text"]:
            example = example.split("Key Facts:")[0].split("Year:")[0]
            example = example.replace("Answer:", "Analysis:")
            instruction = self.random.choice(case_brief_instructions)
            # TODO: We need to split the example into a question and an answer.
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, _BLANK_PROMPT,
                                        example, task_type, jurisdiction)
