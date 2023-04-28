from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class CiviproQuestions(AbstractDataset):

    def __init__(self):
        super().__init__("CiviproQuestions", "https://arxiv.org/abs/2211.02950")

    def get_data(self, instructions: instruction_manager.InstructionManager):

        instruction_bank_generate_questions_from_passage = [
            "Consider these questions about American civil procedure. Given the provided information answer them to the best of your ability.",
            "Here is some information that can help you answer the question. Provide an analysis of the options and then pick the correct answer.",
            "Given a passage of text about Civil Procedure in the United States, generate a question that can be answered by reading the passage.",
            "Generate a CivPro question that can be answered by reading the passage, denote it as \"Question:\", provide an answer as \"Answer\"."
        ]
        instruction_bank_generate_questions_no_passage = [
            "Consider these questions about American civil procedure. Answer them to the best of your ability, first provide an explanation then the answer.",
            "Consider these civil procedure questions. Provide an analysis of the options and then pick the correct answer.",
            "Answer this Civil Procedure question based on law in the United States. Provide an explanation first.",
            "Answer this CivPro question provide an answer as \"Answer\", but first provide an explanation as \"Explanation\"."
        ]
        instruction_bank_generate_questions_no_explanation = [
            "Consider these questions about American civil procedure. Answer them to the best of your ability, DO NOT provide an explanation before giving the answer.",
            "Consider these civil procedure questions. Pick the correct answer.",
            "Given a passage of text about Civil Procedure in the United States, answer the question.",
            "Answer this CivPro question provide an answer as \"Answer\"."
        ]

        df = pd.read_csv(f"{self.raw_data_dir}/civpro_questions_train.csv")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.US
        instruction_language = "en"
        prompt_language = "en"

        questions_dict = defaultdict(dict)

        for idx, row in tqdm(df.iterrows()):
            if row["question"] not in questions_dict:
                questions_dict[row["question"]] = {"choices": []}
            questions_dict[row["question"]]["choices"].append(
                (row["answer"], row["label"], row["analysis"]))
            questions_dict[
                row["question"]]["explanation_passage"] = row["explanation"]
            questions_dict[
                row["question"]]["chain_of_thought"] = row["complete analysis"]

        for question, values in questions_dict.items():
            choices = values["choices"]
            question = ".".join(question.split(".")[1:])
            if len(choices) < 4:
                print(f"Skipping {question} because it has less than 2 choices")
                continue
            # self.random.shuffle(choices)
            lookup = ["A", "B", "C", "D", "E", "F", "G"]
            analysis_string = values[
                'chain_of_thought']  # "\n".join([f"{choice[2]}" for i, choice in enumerate(choices)])
            try:
                choice_string = "\n".join([
                    f"{lookup[i]}. {choice[0]}"
                    for i, choice in enumerate(choices)
                ])
                correct_answer = lookup[[
                    idx for idx, choice in enumerate(choices) if choice[1] == 1
                ][0]]
            except:
                print(f"Skipping {question} because of some problem.")
                continue
            # TODO: should the 'explanation_passage' be part of the instructions?
            instruction_with_passage = self.random.choice(
                instruction_bank_generate_questions_from_passage)
            prompt_with_passage = f"{values['explanation_passage']}\n\nQuestion: {question}\n{choice_string}"
            answer_with_passage = f"Answer: {correct_answer}"
            instruction_no_passage = self.random.choice(
                instruction_bank_generate_questions_no_passage)
            prompt_no_passage = f"Question: {question}\n{choice_string}"
            answer_no_passage = f"Explanation: {analysis_string}\nAnswer: {correct_answer}"
            instruction_no_explanation = self.random.choice(
                instruction_bank_generate_questions_no_explanation)
            prompt_no_explanation = f"Question: {question}\n{choice_string}"
            answer_no_explanation = f"Answer: {correct_answer}"

            for instruction, prompt, answer in zip(
                [
                    instruction_with_passage,
                    instruction_no_passage,
                    instruction_no_explanation,
                ],
                [
                    prompt_with_passage,
                    prompt_no_passage,
                    prompt_no_explanation,
                ],
                [
                    answer_with_passage,
                    answer_no_passage,
                    answer_no_explanation,
                ],
            ):
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction)
