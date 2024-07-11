from collections.abc import Iterator
from datasets import load_dataset

from abstract_dataset import AbstractDataset, DataPoint
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

class LexAbSumm(AbstractDataset):

    def __init__(self):
        super().__init__(
            "LexAbSumm", 
            "https://huggingface.co/datasets/MahmoudAly/LexAbSumm")
        
    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = load_dataset('MahmoudAly/LexAbSumm', split='train')
        task_type = TaskType.SUMMARIZATION
        jurisdiction = Jurisdiction.EU
        prompt_language = "en"
        answer_language = "en"
        instruction_language: str
        
        for example in df:
            subset = "lexabsumm_fact_sum"
            facts = example['facts_source']
            
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Source: {facts}"
            answer = f"Summary: {example['facts_summary']}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset) 

        for example in df:
            subset = "lexabsumm_law_sum"
            facts = example['law_source']
            
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Source: {facts}"
            answer = f"Summary: {example['law_summary']}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset) 

        task_type = TaskType.TEXT_GENERATION

        for example in df:
            subset = "lexabsumm_title"
            summary = example['facts_summary']
            
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Summary: {summary}"
            answer = f"Title: {example['title']}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset)

        for example in df:
            subset = "lexabsumm_subtitle"
            title = example['title']
            summary = example['law_summary']
            
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Title: {title}\n\n" \
                     f"Summary: {summary}"
            answer = f"Subtitle: {example['subtitle']}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, 
                                        answer, task_type, jurisdiction, subset)