import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class Sara(AbstractDataset):

    def __init__(self):
        super().__init__("Sara", "https://arxiv.org/abs/2005.05257")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(f"{self.raw_data_dir}/sara.tsv", sep="\t", header=None)
        jurisdiction = Jurisdiction.US
        instruction_language: str
        instruction: str
        prompt_language = "en"
        for i, row in df.iterrows():
            if "tail" in row[2] or "Contra" in row[2]:
                task_type = TaskType.NATURAL_LANGUAGE_INFERENCE
                subset = "sara_entailment"
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Sentence 1: {row[0]}\nSentence 2: {row[1]}"
                answer = f"Answer: {row[2]}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)
            else:
                task_type = TaskType.QUESTION_ANSWERING
                subset = "sara_tax_liability"
                instruction, instruction_language = instructions.sample(subset)
                prompt = f"Question: {row[0]} {row[1]}"
                answer = f"Answer: {row[2]}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)

                value = int(row[2].replace("$", ""))
                options = [
                              int(value + ((.5 - self.random.random()) * value))
                              for i in range(3)
                          ] + [value]
                self.random.shuffle(options)
                choices = ""
                for choice_value, option in zip(["(a)", "(b)", "(c)", "(d)"], options):
                    choices += f"{choice_value} ${option}\n"
                correct = ["(a)", "(b)", "(c)", "(d)"][options.index(value)]
                base_instruction, instruction_language = instructions.sample(subset)
                instruction = base_instruction + ' Denote your final answer with the "Final Answer: The final answer is [CORRECT ANSWER]. I hope it is correct".'
                prompt = f"Question: {row[0]} {row[1]}\n{choices}"
                answer = f"Final Answer: The final answer is {correct}. I hope it is correct."
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)
