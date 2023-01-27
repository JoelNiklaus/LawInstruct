import json

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class TsccAlqac(AbstractDataset):
    def __init__(self):
        super().__init__("TsccAlqac",
                         "https://github.com/KevinMercury/tscc-dataset-alqac2021/blob/main/tscc_alqac2021_law.json")

    def get_data(self):
        # Thai supreme court case law
        with open("./raw_data/tscc_alqac2021_question.train.json", "r") as f:
            cases = json.loads(f.read())

        with open("./raw_data/tscc_alqac2021_law.json", "r") as f:
            laws = json.loads(f.read())

        laws_dict = {}
        for article in laws[0]['articles']:
            laws_dict[article['id']] = article['text']

        given_facts_output_rules = "Given these facts in the Thai legal system, please output the relevant legal rule(s)"
        instructions_bank = [
            "For the relevant facts, please provide the relevant Thai law(s). Use the rule to determine the court's likely conclusion.",
            f"{given_facts_output_rules} and the court's likely judgement.",
            f"{given_facts_output_rules} and provide the legal conclusion of whether the court is likely to find for or against the defendant.",
            f"{given_facts_output_rules} and provide the legal conclusion of whether the court is likely to find the defendant guilty or not guilty.",
        ]

        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.THAILAND
        prompt_language = "en"
        answer_language = "th"

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
            text = f"{self.random.choice(instructions_bank)}\n\nFacts: {text}\nLaw(s): {laws}\nConclusion: {outcome}"
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)

            # Provide a non-MC version
            outcome_mc1 = ["(a)", "(b)"][case["label"]]
            text = f"{self.random.choice(instructions_bank)}\n\nQuestion: {text} How would the court find?\n(a) For the defendant.\n(b) Against the defendant.\nLaw(s): {laws}\nAnswer: {outcome_mc1}."
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)

            outcome_mc1 = ["(b)", "(a)"][case["label"]]
            text = f"{self.random.choice(instructions_bank)}\n\nQuestion: {text} How would the court find?\n(a) Against the defendant.\n(b) For the defendant.\nLaw(s): {laws}\nAnswer: {outcome_mc1}."
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)