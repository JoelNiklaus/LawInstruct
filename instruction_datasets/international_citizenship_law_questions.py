import pandas as pd

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager


class InternationalCitizenshipLawQuestions(AbstractDataset):

    def __init__(self):
        super().__init__("InternationalCitizenshipLawQuestions",
                         "https://cadmus.eui.eu/handle/1814/73190")

    def get_data(self, instructions: instruction_manager.InstructionManager):
        task_type = TaskType.QUESTION_ANSWERING
        jurisdiction = Jurisdiction.INTERNATIONAL
        instruction_language = "en"
        prompt_language = "en"
        answer_language = "en"

        df_mode_acq = pd.read_csv(
            f"{self.raw_data_dir}/data_v1.0_country-year-mode_acq.csv")
        df_mode_loss = pd.read_csv(
            f"{self.raw_data_dir}/data_v1.0_country-year-mode_loss.csv")
        code_year = pd.read_csv(
            f"{self.raw_data_dir}/data_v1.0_country-year.csv")
        code_dictionary = pd.read_csv(
            f"{self.raw_data_dir}/code_dictionary.csv")

        subset = "international_citizenship_law_questions_mode_acq"
        for idx, row in df_mode_acq.iterrows():
            mode_id = row["mode_id"]
            country = row["country"]
            law_article = row["article"]
            law_article = law_article.strip().replace('\n', ' ')
            specification = row["specification"]
            specification = specification.strip().replace('\n', ' ')
            if specification != "n.a.":
                specification = "The provision applies under the following conditions. " + specification
            else:
                specification = ""
            code_year_spec = code_year[code_year["country"] == country]
            code_year_spec = code_year_spec[f"{mode_id.strip()}_bin"].values[0]
            if code_year_spec == 99:
                code_year_spec = 0
            code_year_spec_answer = ["No.", "Yes."][code_year_spec]
            q = code_dictionary[code_dictionary["Mode ID"] ==
                                mode_id.strip()]["Focus"].values[0]

            prompt = f"Q: Consider the country of {country.strip()}. {q.strip()}"
            if "No provision" in law_article:
                answer = f"A: {code_year_spec_answer} This is not covered in any provision."
            else:
                answer = f"A: {code_year_spec_answer} This is covered in: {law_article}. {specification}".strip(
                )

            instruction, instruction_language = instructions.sample(subset)

            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction, subset)

        subset = "international_citizenship_law_questions_mode_loss"
        for idx, row in df_mode_loss.iterrows():
            mode_id = row["mode_id"]
            country = row["country"]
            law_article = row["article"]
            law_article = law_article.strip().replace('\n', ' ')

            specification = row["specification"]
            specification = specification.strip().replace('\n', ' ')
            if specification != "n.a.":
                specification = "The provision applies under the following conditions. " + specification
            else:
                specification = ""

            code_year_spec = code_year[code_year["country"] == country]
            code_year_spec = code_year_spec[f"{mode_id.strip()}_bin"].values[0]
            if code_year_spec == 99:
                code_year_spec = 0
            code_year_spec_answer = ["No.", "Yes."][code_year_spec]
            q = code_dictionary[code_dictionary["Mode ID"] ==
                                mode_id.strip()]["Focus"].values[0]
            prompt = f"Q: Consider the country of {country.strip()}. {q.strip()}"
            if "No provision" in law_article:
                answer = f"A: {code_year_spec_answer} This is not covered in any provision."
            else:
                answer = f"A: {code_year_spec_answer} This is covered in: {law_article}. {specification}".strip(
                )

            instruction, instruction_language = instructions.sample(subset)
            
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt, answer, task_type,
                                        jurisdiction, subset)
