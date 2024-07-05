from collections.abc import Iterator
from collections.abc import Sequence
from datasets import load_dataset
from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


def extract_negation_cues(data: dict) -> list[str]:
    negation_cues = []
    for span in data['spans']:
        if span['label'] == 'CUE':
            cue_text = data['text'][span['start']:span['end']]
            negation_cues.append(cue_text)
    return negation_cues

def extract_negation_scopes(data: dict) -> list[str]:
    text = data['text']
    spans = data['spans']
    results = []
    current_scope_before = None
    current_cue = None
    for span in spans:
        if span['label'] == 'SCOPE' and current_cue is None:
            current_scope_before = text[span['start']:span['end']]
        elif span['label'] == 'CUE':
            current_cue = text[span['start']:span['end']]
            scope_after_cue = ""
            for after_span in spans:
                if after_span['label'] == 'SCOPE' and after_span['start'] > span['end']:
                    scope_after_cue = text[after_span['start']:after_span['end']]
                    break
            result = {
                'scope_before_cue': current_scope_before,
                'cue': current_cue,
                'scope_after_cue': scope_after_cue
            }
            results.append(result)
            current_scope_before = scope_after_cue
            current_cue = None

    return results 

def build_answer_cue(negations: Sequence[str]) -> str:
    markdown_list = ""
    for negation in negations:
        markdown_list += "- " + negation + " \n"
    return markdown_list 

def build_answer_scopes(data: Sequence[str]) -> str: 
    markdown_list = ""
    for item in data:
        if item['cue'] == None:
            item['cue'] = "None"
        if item['scope_before_cue'] == None:
            item['scope_before_cue'] = "None"  
        if item['scope_after_cue'] == None:
            item['scope_after_cue'] = "None"         
        markdown_list += "- cue: " + item['cue'] + \
                         ", scope before cue: " + item['scope_before_cue']+ \
                         ", scope after cue: " + item['scope_after_cue'] + " \n"
    return markdown_list 



class MultiLegalNegation(AbstractDataset):

    def __init__(self):
        super().__init__("MultiLegalNegation", 
                         "https://huggingface.co/datasets/rcds/MultiLegalNeg")
        

    def get_data(self, instructions: instruction_manager.InstructionManager):
        subsets = [
        {"name": "de", "jurisdiction": Jurisdiction.GERMANY, "language": "de"},
        {"name": "fr", "jurisdiction": Jurisdiction.FRANCE, "language": "fr"},
        {"name": "it", "jurisdiction": Jurisdiction.ITALY, "language": "it"},
        {"name": "swiss", "jurisdiction": Jurisdiction.SWITZERLAND, "language": "de"},
        ]

        task_type = TaskType.TEXT_CLASSIFICATION

        for subds in subsets:
            print(f"Processing subset: {subds['name']}")
            ds= load_dataset("rcds/MultiLegalNeg", subds['name'], split ='train')
            for example in ds:
                subset = 'multi_legal_cue_detection'+'_'+subds['name']
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Text: {example['text']}"
                negation_cues = extract_negation_cues(example)
                answer = f"Extracted negations: {build_answer_cue(negation_cues)}"
                yield self.build_data_point(instruction_language, subds['language'],
                                            subds['language'], instruction, prompt, 
                                            answer, task_type, subds['jurisdiction'], subset)
            for example in ds:
                subset = 'multi_legal_scope_detection'+'_'+subds['name']
                instruction, instruction_language = instructions.sample(subset)
                negation_cues = extract_negation_cues(example)
                prompt = f"Text: {example['text']}\n" \
                        f"Negation cues: {build_answer_cue(negation_cues)}" 
                negation_scopes = extract_negation_scopes(example)  
                answer = f"Scopes for given cues: {build_answer_scopes(negation_scopes)}"
                yield self.build_data_point(instruction_language, subds['language'],
                                            subds['language'], instruction, prompt, 
                                            answer, task_type, subds['jurisdiction'], subset)       
                
