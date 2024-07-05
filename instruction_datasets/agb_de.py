from collections.abc import Iterator
from datasets import load_dataset

from abstract_dataset import AbstractDataset, DataPoint
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

TOPIC_LIST = "age, applicability, applicableLaw, arbitration, changes, codeOfConduct, conclusionOfContract, delivery, description, disposal, intellectualProperty, liability, party, payment, personalData, placeOfJurisdiction, prices, retentionOfTitle, severability, textStorage, warranty, withdrawal"

class AGBDE(AbstractDataset):

    def __init__(self):
        super().__init__(
            "AGBDE", 
            "https://huggingface.co/datasets/d4br4/agb-de")
        
    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('d4br4/agb-de', split='train')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.GERMANY
        prompt_language = "de"
        answer_language = "en"
        instruction_language: str

        label_mapping = {0: 'valid', 1: 'potentially void'}
        
        for example in df:
            subset = "agb_de_judgment"
            title = example['title']
            text = example['text']
            clause = (title or '') + (text or '')
            
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Clause: {clause}"
            answer = f"This clause is {label_mapping[example['label']]}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset) 

        task_type = TaskType.MULTIPLE_CHOICE

        for example in df:
            subset = "agb_de_mc"
            title = example['title']
            text = example['text']
            clause = (title or '') + (text or '')
            topics = ', '.join(example['topics'])  

            
            instruction, instruction_language = instructions.sample(subset)
            full_instruction = f"{instruction} Please use only the following topics: {TOPIC_LIST}"
            prompt = f"Clause: {clause}"
            answer = f"Topic(s): {topics}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, full_instruction, prompt, 
                                        answer, task_type, jurisdiction, subset) 
