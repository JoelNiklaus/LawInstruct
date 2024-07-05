import copy
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
    num = 1
    markdown_list = ""
    for citation in citations:
        markdown_list += "- <citation" + str(num) + "> : " + citation + " \n"
        num += 1
    return markdown_list      

def extract_citations(consideration: Sequence[str], ner_labels: Sequence[int]) -> list:
    ner_labels_np = np.array(ner_labels)
    citations_idx = np.nonzero(ner_labels_np)
    citations = []
    prev_num = None
    citation = ""
    for idx in citations_idx[0]:
        if prev_num is not None and idx != prev_num + 1:
            citations.append(citation)
            citation = consideration[idx]
        else:
            citation += consideration[idx] if citation == "" or consideration[idx] == '.' else " " + consideration[idx]
        prev_num = idx
    citations.append(citation)
    return citations  
 
def build_sentence(considerations: Sequence[str]) -> str:
    no_space_before = [')', ']', '{', '}', ';', ':', "'", '"', ',', '>', '.','?', '_', '-', '/']
    no_space_after = ['(', '[', '{','<' "'", '"','_', '-', '/']
    sentence = ""
    previous_word = None
    for word in considerations:
        if previous_word in no_space_after or word in no_space_before:
            sentence += word
        else:
            sentence += " " + word
        previous_word = word
    return sentence 

def mask_citations(consideration: Sequence[str], ner_labels: Sequence[int]) -> list:
    masked_consideration = copy.deepcopy(consideration)
    ner_labels_np = np.array(ner_labels)
    citations_idx = np.nonzero(ner_labels_np)
    prev_num = None
    num = 1
    for idx in citations_idx[0]:
        if prev_num is None or idx != prev_num + 1:
            masked_consideration[idx] =  '<citation' + str(num) + '>'
            num += 1
        else:
            masked_consideration[idx] = None  
        prev_num = idx 
    masked_consideration = [token for token in masked_consideration if token is not None]  
    return masked_consideration  
           
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

           



