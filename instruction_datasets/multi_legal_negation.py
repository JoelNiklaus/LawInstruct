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
            cue = data['text'][span['start']:span['end']]
            negation_cues.append(cue)
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

def build_answer_scopes(data: Sequence[str]) -> str:
    scope_before_cue = data['scope_before_cue']
    scope_after_cue = data['scope_after_cue']
    if scope_before_cue == None:
        scope_before_cue = ''
    if scope_after_cue == None: 
        scope_after_cue = ''
    answer = 'scope before cue: ' + scope_before_cue +\
            ', scope after cue: '  + scope_after_cue
    return answer



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
        subset1 = 'multi_legal_cue_detection'
        subset2 = 'multi_legal_scope_detection'

        for subds in subsets:
            ds = load_dataset("rcds/MultiLegalNeg", subds['name'], split='train')

            for example in ds:
                instruction, instruction_language = instructions.sample(subset1)
                prompt = f"Text: {example['text']}"
                negation_cues = extract_negation_cues(example)
                for neg in negation_cues:
                        answer = f"Extracted negation cue: {neg}"
                        yield self.build_data_point(instruction_language, subds['language'],
                                                subds['language'], instruction, prompt, 
                                                answer, task_type, subds['jurisdiction'], subset1)
        
        for subds in subsets: 
            ds = load_dataset("rcds/MultiLegalNeg", subds['name'], split='train')

            for example in ds:
                instruction, instruction_language = instructions.sample(subset2)
                negation_scopes = extract_negation_scopes(example)
                for scope in negation_scopes:
                    prompt = f"Text: {example['text']}\n" \
                            f"Negation cue: {scope['cue']}"
                    answer = f"Scopes for given cue: {build_answer_scopes(scope)}"
                    yield self.build_data_point(instruction_language, subds['language'],
                                                subds['language'], instruction, prompt, 
                                                answer, task_type, subds['jurisdiction'], subset2)    
                
