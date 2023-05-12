import json
from typing import Final

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class TsccAlqac(AbstractDataset):

    def __init__(self):
        super().__init__(
            "TsccAlqac",
            "https://github.com/KevinMercury/tscc-dataset-alqac2021/blob/main/tscc_alqac2021_law.json"
        )

    def get_data(self, instructions: instruction_manager.InstructionManager):
        # Thai supreme court case law
        with open(f"{self.raw_data_dir}/tscc_alqac2021_question.train.json",
                  "r") as f:
            cases = json.loads(f.read())

        with open(f"{self.raw_data_dir}/tscc_alqac2021_law.json", "r") as f:
            laws = json.loads(f.read())

        laws_dict = {}
        for article in laws[0]['articles']:
            laws_dict[article['id']] = article['text']

        subset = "tscc_alqac_question_answering"

        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.THAILAND
        prompt_language = "en"
        answer_language = "th"  # TODO: is this correct? Looks like `outcome` is English.

        for case in cases:
            text = case["text"]
            relevant_articles = []
            for article in case["relevant_articles"]:
                law_text = laws_dict[article['article_id']]
                relevant_articles.append(law_text)

            # Provide a MC version for the judgement
            if self.random.random() > .5:
                outcome = f"The court would likely find the defendant{'' if case['label'] == 1 else ' not'} guilty."
            else:
                outcome = f"The court would rule {'against' if case['label'] == 1 else 'for'} the defendant."
            laws = '\n'.join(relevant_articles)
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Facts: {text}\nLaw(s): {laws}"
            answer = f'Conclusion: {outcome}'
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

            # Provide a non-MC version
            outcome_mc1 = ["(a)", "(b)"][case["label"]]
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {text} How would the court find?\n" \
                     f"(a) For the defendant.\n(b) Against the defendant.\nLaw(s): {laws}"
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

            outcome_mc1 = ["(b)", "(a)"][case["label"]]
            instruction, instruction_language = instructions.sample(subset)
            prompt = f"Question: {text} How would the court find?\n" \
                     f"(a) Against the defendant.\n(b) For the defendant.\nLaw(s): {laws}"
            answer = f"Answer: {outcome_mc1}."
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)
