from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

class KeyphraseGenerationEU(AbstractDataset): 
    
    def get_data(self, instructions: instruction_manager.InstructionManager): 
        ds = load_dataset("NCube/europa", split =' train')
        task_type = TaskType.SUMMARIZATION
        jurisdiction = Jurisdiction.EU
        instruction_language: str
        for example in ds:
            subset = 'keyphrase_generation_eu'
            instruction, instruction_language = instructions.sample(subset)
            promt = f"Judgment transcription: {example['input_text']}"
            answer = f"Keyphrases : {example['keyphrases']}"
            yield self.build_data_point(instruction_language, example['lang'],
                                        example['lang'], instruction,
                                        promt, answer, task_type, 
                                        jurisdiction, subset)