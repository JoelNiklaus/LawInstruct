from collections.abc import Iterator
from datasets import load_dataset

from abstract_dataset import AbstractDataset, DataPoint
from enums import Jurisdiction
from enums import TaskType
import instruction_manager

class GermanLaymenQA(AbstractDataset):

    def __init__(self):
        super().__init__(
            "GermanLaymenQA", 
            "https://huggingface.co/datasets/ViolaCamille/GerLayQA")
        
    def get_data(self, instructions: instruction_manager.InstructionManager):

        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.GERMANY
        prompt_language = "de"
        answer_language = "de"

        sub_datasets = ["german_laymen_bgb_qa", "german_laymen_stgb_qa",
                         "german_laymen_zpo_qa"]
        
        for sub_dataset in sub_datasets: 
            df = load_dataset('ViolaCamille/GerLayQA', sub_dataset, split = 'train')
            for example in df:
                subset = sub_dataset
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Q: {example['Question_text']}"
                answer = f"A: {example['Answer_text']}"
                yield self.build_data_point(instruction_language, prompt_language,
                                            answer_language, instruction, prompt, 
                                            answer, task_type, jurisdiction, subset) 


