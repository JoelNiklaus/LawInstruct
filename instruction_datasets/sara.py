import pandas as pd
from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class Sara(AbstractDataset):
    def __init__(self):
        super().__init__("Sara", "https://arxiv.org/abs/2005.05257")

    def get_data(self):
        print("############################")
        print("########## Sara ###########")
        print("############################")
        df = pd.read_csv("raw_data/sara.tsv", sep="\t", header=None)
        jurisdiction = JURISDICTION.US
        prompt_language = "en"
        entailment_instruction_bank = [
            "Consider the following US Tax scenario. Does the first fact entail the second fact?",
            "Are these two sentences entailed or contradicting?",
            "Respond entailement or contradiction to these two sentences."]
        tax_liability_instruction_bank = ["Consider the following US Tax scenario and answer the question.",
                                          "Consider the following scenario. Calculate the right amount of tax liablity and answer the question."]
        for i, row in df.iterrows():
            if "tail" in row[2] or "Contra" in row[2]:
                task_type = TASK_TYPE.NATURAL_LANGUAGE_INFERENCE
                datapoint = f"{self.random.choice(entailment_instruction_bank)}\n\nSentence 1: {row[0]}\nSentence 2: {row[1]}\nAnswer: {row[2]}"
                yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)
            else:
                task_type = TASK_TYPE.QUESTION_ANSWERING
                datapoint = f"{self.random.choice(tax_liability_instruction_bank)}\n\nQuestion: {row[0]} {row[1]}\nAnswer: {row[2]}"
                yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)

                value = int(row[2].replace("$", ""))
                options = [int(value + ((.5 - self.random.random()) * value)) for i in range(3)] + [value]
                self.random.shuffle(options)
                choices = ""
                for choice_value, option in zip(["(a)", "(b)", "(c)", "(d)"], options):
                    choices += f"{choice_value} ${option}\n"
                correct = ["(a)", "(b)", "(c)", "(d)"][options.index(value)]
                datapoint = f"{self.random.choice(tax_liability_instruction_bank)} Denote your final answer with the \"Final Answer: The final answer is [CORRECT ANSWER]. I hope it is correct\".\n\nQuestion: {row[0]} {row[1]}\n{choices}\n\nFinal Answer: The final answer is {correct}. I hope it is correct."
                yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)