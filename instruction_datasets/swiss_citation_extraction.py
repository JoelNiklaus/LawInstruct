from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
from collections.abc import Sequence
import numpy as np 
import instruction_manager

from datasets import load_dataset

def build_answer_extraction(citations: Sequence[str]) -> str:
    markdown_list = ""
    for citation in citations:
        markdown_list += "- " + citation + " \n"
    return markdown_list 

def build_answer_prediction(citations: Sequence[str]) -> str: 
    markdown_list = ""
    for citation in citations:
        markdown_list += "- : " + citation + " \n"
    return markdown_list    

def extract_citations(consideration: Sequence[str], ner_labels: Sequence[int]) -> list:
    ner_labels_np = np.array(ner_labels)
    citations_idx = np.nonzero(ner_labels_np)
    citations = []
    prev_num = None
    citation = ""
    for idx in citations_idx[0]:
        if prev_num is None or idx == prev_num + 1:
            citation += consideration[idx] + " "   
        else:
            citations.append(citation)  
            citation = ""
            citation += consideration[idx] + " "
        prev_num = idx
    citations.append(citation)    
    return citations    

def build_sentence(input: Sequence[str]) -> str: 
    consideration = " ".join(input)
    return consideration 


def mask_citations(consideration: Sequence[str], ner_labels: Sequence[int]) -> list:
    ner_labels_np = np.array(ner_labels)
    citations_idx = np.nonzero(ner_labels_np)
    for idx in citations_idx[0]:
        consideration[idx] = ',,,'
    return consideration    
           
class SwissCitationExtraction(AbstractDataset):

    def __init__(self):
        super().__init__(
            "SwissCitationExtraction",
            "https://huggingface.co/datasets/rcds/swiss_citation_extraction/viewer/original/test"
        )

    def get_data(self, instructions: instruction_manager.InstructionManager):

        ds = load_dataset("rcds/swiss_citation_extraction", "original", split='test')
        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.SWITZERLAND
        instruction_language = str

        for example in ds: 
            subset = "swiss_citation_extraction"
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Consideration: {build_sentence(example['considerations'])}"
            citations = extract_citations(example['considerations'], example['NER_labels'])
            answer = f"Citations: {build_answer_extraction(citations)}"
            yield self.build_data_point(instruction_language, example['language'],
                                        example['language'], instruction, 
                                        prompt, answer, task_type, 
                                        jurisdiction, subset)
            
        task_type = TaskType.TEXT_GENERATION

        for example in ds: 
            subset = "swiss_citation_prediction"
            instruction, instruction_language = instructions.sample(subset)
            masked_consideration = mask_citations(example['considerations'], example['NER_labels'])
            prompt = f"Masked Consideration: {build_sentence(masked_consideration)}"
            citations = extract_citations(example['considerations'], example['NER_labels'])
            answer = f"Citations: {build_answer_prediction(citations)}"
            yield self.build_data_point(instruction_language, example['language'],
                                        example['language'], instruction, 
                                        prompt, answer, task_type, 
                                        jurisdiction, subset)

           



