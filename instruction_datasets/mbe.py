import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class MBE(AbstractDataset):

    def __init__(self):
        # TODO do we have an url for the source here?: Lucia's working paper
        super().__init__("MBE", "MBE")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(f"{self.raw_data_dir}/mbe_train.csv")
        task_type = TaskType.MULTIPLE_CHOICE
        jurisdiction = Jurisdiction.US
        instruction_language = "en"
        prompt_language = "en"

        instructions_examples = [
            "Answer these legal questions. Use American Law. Please explain your thought process and then answer the question.",
            "Answer these U.S. Multistate Bar Exam questions. Please provide an explanation first.",
            "Pick the most correct option considering U.S. Law. Explain your answer first."
        ]
        instruction_bank_subject = [
            "What subject of U.S. law is this question about? Pick one from: TORTS, CONTRACTS, CRIM. LAW, EVIDENCE, CONST. LAW, REAL PROP.",
            "What area of American law is this question about? Pick one from: TORTS, CONTRACTS, CRIM. LAW, EVIDENCE, CONST. LAW, REAL PROP."
        ]
        instruction_bank_subject_generation = [
            "Generate a bar exam multiple choice question, along with an explanation and answer, for the following subject: ",
            "Generate an MBE MC question for "
        ]
        for idx, row in df.iterrows():
            # source_year = row["Source"].split("MBE-")[1].split("-")[0]
            if isinstance(row['Prompt'], str) and row['Prompt'].strip(
            ) != "" and row['Prompt'].strip() != "nan":
                question = row['Prompt']
            else:
                question = ""
            question += f" {row['Question']}"
            question = question.strip()
            choices = [
                row["Choice A"], row["Choice B"], row["Choice C"],
                row["Choice D"]
            ]
            answer = row["Answer"]
            subject = row["Subject"]
            positive_passage = row["Positive Passage"]
            datapoint = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                lookup = ["A", "B", "C", "D"]
                datapoint += f"{lookup[i]}. {choice}\n"
            data_no_answer = datapoint
            answer = f"Explanation: {positive_passage}\nAnswer: {answer}"
            data_with_answer = data_no_answer + answer
            # if source_year.strip() != "" and int(source_year) > 1950 and int(source_year) < 2023:
            #     source_year_string = f" Consider only the law and relevant cases before {source_year}."
            # else:
            #     source_year_string = ""
            instruction = self.random.choice(instructions_examples)
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, data_no_answer,
                                        answer, task_type, jurisdiction)

            if isinstance(subject, str) and subject.strip() != "":
                # Datapoint with subject.
                instruction = self.random.choice(instruction_bank_subject)
                prompt = data_no_answer
                answer = f"Subject: {subject}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction)

                # Datapoint for generation with subject.
                instruction = self.random.choice(
                    instruction_bank_subject_generation) + subject
                _BLANK_PROMPT = ""  # TODO: what would the prompt be here?
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            _BLANK_PROMPT, data_with_answer,
                                            task_type, jurisdiction)
