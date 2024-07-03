from collections.abc import Sequence
from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


def build_answer(keyphrases_list: Sequence[Sequence[str]]) ->str:
    markdown_list = ""
    for keyphrases in keyphrases_list:
        for keyphrase in keyphrases:
            markdown_list += "- " + keyphrase + "\n"
    return markdown_list 

class KeyphraseGenerationEU(AbstractDataset): 

    def __init__(self):
        super().__init__(
            "KeyphraseGenerationEU",
            "https://huggingface.co/datasets/NCube/europa"
        )
    
    def get_data(self, instructions: instruction_manager.InstructionManager): 
        ds = load_dataset("NCube/europa", split ='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.EU
        instruction_language: str
        for example in ds:
            subset = 'keyphrase_generation_eu'
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Judgment transcription: {example['input_text']}"
            answer = f"Keyphrases: {build_answer(example['keyphrases'])}"
            yield self.build_data_point(instruction_language, example['lang'],
                                        example['lang'], instruction,
                                        prompt, answer, task_type, 
                                        jurisdiction, subset)