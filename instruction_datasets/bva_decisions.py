import json
import os

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType


class BVADecisions(AbstractDataset):

    def __init__(self):
        super().__init__("BVADecisions",
                         "https://github.com/LLTLab/VetClaims-JSON")

    def get_data(self):

        json_files = [
            f"{self.raw_data_dir}/VetClaims-JSON/BVA Decisions JSON Format/{pos_json}"
            for pos_json in os.listdir(
                f"{self.raw_data_dir}/VetClaims-JSON/BVA Decisions JSON Format")
            if pos_json.endswith('.json')
        ]
        json_files.extend([
            f"{self.raw_data_dir}/VetClaims-JSON/BVA Decisions JSON Format +25/{pos_json}"
            for pos_json in os.listdir(
                f"{self.raw_data_dir}/VetClaims-JSON/BVA Decisions JSON Format +25"
            )
            if pos_json.endswith('.json')
        ])
        sentences = []
        rule_trees = []
        for json_f in json_files:
            with open(json_f, "r") as f:
                x = json.loads(f.read())
                sentences.extend(x["sentences"])
                rule_trees.append(x["ruleTree"])
        instruction_bank = [
            "Label the sentence according to its rhetorical role in a legal argument.",
            "Please label the sentence as according to its role as either a FindingSentence, a ReasoningSentence, a LegalRuleSentence, a CitationSentence, or an EvidenceSentence. If it is none of these, mark it as just Sentence.",
            "Please label the following according to one of these categories.\n\tFindingSentence. A finding sentence is a sentence that primarily states an authoritative finding, conclusion or determination of the trier of fact – a decision made “as a matter of fact” instead of \"as a matter of law.\"\n\tReasoningSentence. A reasoning sentence is a sentence that primarily reports the trier of fact’s reasoning based on the evidence, or evaluation of the probative value of the evidence, in making the findings of fact.\n\tEvidenceSentence. An evidence sentence is a sentence that primarily states the content of the testimony of a witness, states the content of documents introduced into evidence, or describes other evidence.\n\tLegalRuleSentence. A legal-rule sentence is a sentence that primarily states one or more legal rules in the abstract, without stating whether the conditions of the rule(s) are satisfied in the case being decided.\n\tCitationSentence. A citation sentence is a sentence whose primary function is to reference legal authorities or other materials, and which usually contains standard notation that encodes useful information about the cited source.\n\tSentence. All other sentences."
        ]

        def turn_rule_tree_to_text(tree, n=0):
            op = ""
            if 'operation' in tree:
                op = tree['operation']
            elif 'inferenceRelation' in tree:
                op = tree['inferenceRelation']

            if "label" in tree:
                text = f"{op} {tree['label']}"
            else:
                text = f"{op} {tree['name']}"

            if "nodes" not in tree and "children" not in tree:
                return text
            else:
                if "nodes" in tree:
                    children = tree["nodes"]
                else:
                    children = tree["children"]
                if isinstance(children, dict):
                    node_text = f"{''.join(['    '] * n)}{children['inferenceRelation']} {children['name']}"
                else:
                    node_text = "\n".join([
                        f"{''.join(['    '] * n)}{turn_rule_tree_to_text(node, n + 1)}"
                        for node in children
                    ])
                return f"{text}\n{node_text}"

        task_type = TaskType.TEXT_CLASSIFICATION
        jurisdiction = Jurisdiction.US
        instruction_language = "en"
        prompt_language = "en"

        for sentence in sentences:
            if 'rhetClass' in sentence:
                role = sentence['rhetClass']
            else:
                role = ",".join(sentence['rhetRole'])
            instruction = self.random.choice(instruction_bank)
            prompt = f"Sentence: {sentence['text'].strip()}"
            answer = f"Rhetorical Role: {role.strip()}"
            yield self.build_data_point(instruction_language, prompt_language, "en", instruction,
                                        prompt, answer, task_type, jurisdiction)

        task_type = TaskType.QUESTION_ANSWERING
        instruction_bank = [
            "Take the following sentence, name all the rules that would be required to back up the claim. Do so in tree format with logical operators like AND and OR.",
            "Name all the rules that would be required to back up the claim."
        ]
        known_data = []
        for sentence, tree_rule in zip(sentences, rule_trees):
            tree_rule = turn_rule_tree_to_text(tree_rule)
            instruction = self.random.choice(instruction_bank)
            sentence = f"Sentence: {sentence['text'].strip()}"
            answer = f"Rules: {tree_rule.strip()}"
            if answer not in known_data:
                yield self.build_data_point(instruction_language, prompt_language, "en", instruction,
                                            sentence, answer, task_type, jurisdiction)
                known_data.append(answer)
