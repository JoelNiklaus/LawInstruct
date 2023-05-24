import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager
import multiple_choice


class MBE(AbstractDataset):

    def __init__(self):
        # TODO do we have an url for the source here?: Lucia's working paper
        super().__init__("MBE", "MBE")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        df = pd.read_csv(f"{self.raw_data_dir}/mbe_train.csv")
        task_type = TaskType.MULTIPLE_CHOICE
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"

        # TODO bring instruction bank to json, paraphrase and translate
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
            lookup = multiple_choice.sample_markers_for_options(choices)
            for i, choice in enumerate(choices):
                datapoint += f"{lookup[i]}. {choice}\n"
            data_no_answer = datapoint
            answer = f"Explanation: {positive_passage}\nAnswer: {answer}"
            data_with_answer = data_no_answer + answer
            # if source_year.strip() != "" and int(source_year) > 1950 and int(source_year) < 2023:
            #     source_year_string = f" Consider only the law and relevant cases before {source_year}."
            # else:
            #     source_year_string = ""
            subset = 'mbe_examples'
            instruction, instruction_language = instructions.sample(subset)
            yield self.build_data_point(instruction_language, prompt_language,
                                        "en", instruction, data_no_answer,
                                        answer, task_type, jurisdiction, subset)

            if isinstance(subject, str) and subject.strip() != "":
                # Datapoint with subject.
                subset = 'mbe_subject'
                instruction, instruction_language = instructions.sample(subset)
                prompt = data_no_answer
                answer = f"Subject: {subject}"
                yield self.build_data_point(instruction_language,
                                            prompt_language, "en", instruction,
                                            prompt, answer, task_type,
                                            jurisdiction, subset)

                # Datapoint for generation with subject.
                instruction = self.random.choice(
                    instruction_bank_subject_generation) + subject
                _BLANK_PROMPT = ""  # TODO: what would the prompt be here?
                yield self.build_data_point('en',
                                            prompt_language, "en", instruction,
                                            _BLANK_PROMPT, data_with_answer,
                                            task_type, jurisdiction, "mbe_subject_generation")
