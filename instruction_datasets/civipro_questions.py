from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager
import multiple_choice


class CiviproQuestions(AbstractDataset):

    def __init__(self):
        super().__init__("CiviproQuestions", "https://arxiv.org/abs/2211.02950")

    def get_data(self, instructions: instruction_manager.InstructionManager):

        df = pd.read_csv(f"{self.raw_data_dir}/civpro_questions_train.csv")
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.US
        instruction_language: str
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
                print(f"Skipping {question} because it has less than 4 choices")
                continue
            # self.random.shuffle(choices)
            lookup = multiple_choice.sample_markers_for_options(choices)
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
            subset_with_passage = "civipro_questions_generate_from_passage"
            instruction_with_passage, instruction_with_passage_language = instructions.sample(subset_with_passage)
            prompt_with_passage = f"{values['explanation_passage']}\n\nQuestion: {question}\n{choice_string}"
            answer_with_passage = f"Answer: {correct_answer}"

            subset_no_passage = "civipro_questions_generate_no_passage"
            instruction_no_passage, instruction_no_passage_language = instructions.sample(subset_no_passage)
            prompt_no_passage = f"Question: {question}\n{choice_string}"
            answer_no_passage = f"Explanation: {analysis_string}\nAnswer: {correct_answer}"

            subset_no_explanation = "civipro_questions_no_explanation"
            instruction_no_explanation, instruction_no_explanation_language = instructions.sample(subset_no_explanation)
            prompt_no_explanation = f"Question: {question}\n{choice_string}"
            answer_no_explanation = f"Answer: {correct_answer}"

            instructions = [instruction_with_passage, instruction_no_passage, instruction_no_explanation]
            instruction_langs = [instruction_with_passage_language, instruction_no_passage_language,
                                 instruction_no_explanation_language]
            prompts = [prompt_with_passage, prompt_no_passage, prompt_no_explanation]
            answers = [answer_with_passage, answer_no_passage, answer_no_explanation]
            subsets = [subset_with_passage, subset_no_passage, subset_no_explanation]

            for subset in subsets:
                for instruction, instruction_language, prompt, answer in zip(
                        instructions, instruction_langs, prompts, answers
                ):
                    yield self.build_data_point(instruction_language,
                                                prompt_language, "en", instruction,
                                                prompt, answer, task_type,
                                                jurisdiction, subset)
