from typing import List

from datasets import load_dataset
from tqdm import tqdm

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class ProfessionalLaw(AbstractDataset):

    def __init__(self):
        super().__init__("ProfessionalLaw", "https://arxiv.org/abs/2009.03300")
        self.filter_out_mmmlu = True

    def get_data(self, instructions: instruction_manager.InstructionManager):
        if self.filter_out_mmmlu:
            return  # This dataset is part of mmmlu, just ignore it

        # The first 1200 are extra bar exam questions, not sure if we want to keep these in
        df = load_dataset("hendrycks_test",
                          "professional_law",
                          split="auxiliary_train").select(range(1200))
        task_type = TaskType.MULTIPLE_CHOICE
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        def shuffle_choices(choices: List[str], answer: int):
            x = list(enumerate(choices))
            self.random.shuffle(x)
            indices, choices = zip(*x)
            answer = indices.index(answer)
            return choices, answer

        for i, (this_question, this_choices, this_answer) in tqdm(
                enumerate(zip(df["question"], df["choices"], df["answer"])),
                total=len(df)):
            prompt_samples = df.select(
                self.random.sample(
                    list(range(0, i)) + list(range(i + 1, len(df))), 3))
            prompt = ""
            for j, (prompt_question, prompt_choices,
                    prompt_answer) in enumerate(
                        zip(prompt_samples["question"],
                            prompt_samples["choices"],
                            prompt_samples["answer"])):
                prompt += f"Question: {prompt_question}\n"
                lookup = ["(a)", "(b)", "(c)", "(d)"]
                prompt_choices, prompt_answer = shuffle_choices(
                    prompt_choices, prompt_answer)
                for i, choice in enumerate(prompt_choices):
                    prompt += f"{lookup[i]} {choice}\n"
                prompt += (f"The Final Answer: {lookup[prompt_answer]}\n\n")
                prompt += "###\n\n"

            cur_question = prompt
            cur_question += f"Question: {this_question}\n"
            lookup: list[str] = ["(a)", "(b)", "(c)", "(d)"]
            for i, choice in enumerate(this_choices):
                cur_question += f"{lookup[i]} {choice}\n"
            cur_question.rstrip("\n")  # Remove trailing newline

            cur_answer = f"The Final Answer: {lookup[this_answer]}"

            subset = "professional_law_examples"
            instruction, instruction_language = instructions.sample(subset)
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, cur_question,
                                        cur_answer, task_type, jurisdiction, subset)

            subset = "professional_law_zero_shot"
            instruction_zero_shot, instruction_language = instructions.sample(subset)
            question_zero_shot = cur_question.split("###")[-1].strip()
            answer_zero_shot = cur_answer.replace("The Final Answer: ",
                                                  "Answer: ")
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction_zero_shot,
                                        question_zero_shot, answer_zero_shot,
                                        task_type, jurisdiction, subset)
