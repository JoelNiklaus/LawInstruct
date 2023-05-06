import json
import os

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class BVADecisions(AbstractDataset):

    def __init__(self):
        super().__init__("BVADecisions",
                         "https://github.com/LLTLab/VetClaims-JSON")

    def get_data(self, instructions: instruction_manager.InstructionManager):

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
        instruction_language: str
        prompt_language = "en"

        for sentence in sentences:
            if 'rhetClass' in sentence:
                role = sentence['rhetClass']
            else:
                role = ",".join(sentence['rhetRole'])
            instruction, instruction_language = instructions.sample("bva_decisions_label")
            prompt = f"Sentence: {sentence['text'].strip()}"
            answer = f"Rhetorical Role: {role.strip()}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, prompt, answer,
                                        task_type, jurisdiction)

        task_type = TaskType.QUESTION_ANSWERING
        known_data = []
        for sentence, tree_rule in zip(sentences, rule_trees):
            tree_rule = turn_rule_tree_to_text(tree_rule)
            instruction, instruction_language = instructions.sample("bva_decisions_qa")
            sentence = f"Sentence: {sentence['text'].strip()}"
            answer = f"Rules: {tree_rule.strip()}"
            if answer not in known_data:
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            sentence, answer, task_type,
                                            jurisdiction)
                known_data.append(answer)
