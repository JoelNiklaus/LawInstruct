import pandas as pd

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class MBE(AbstractDataset):
    def __init__(self):
        # TODO do we have an url for the source here?: Lucia's working paper
        super().__init__("MBE", "MBE")

    def get_data(self):
        df = pd.read_csv(f"{self.raw_data_dir}/mbe_train.csv")
        task_type = TASK_TYPE.MULTIPLE_CHOICE
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        instructions_examples = [
            "Answer these legal questions. Use American Law. Please explain your thought process and then answer the question.",
            "Answer these U.S. Multistate Bar Exam questions. Please provide an explanation first.",
            "Pick the most correct option considering U.S. Law. Explain your answer first."]
        instruction_bank_subject = [
            "What subject of U.S. law is this question about? Pick one from: TORTS, CONTRACTS, CRIM. LAW, EVIDENCE, CONST. LAW, REAL PROP.",
            "What area of American law is this question about? Pick one from: TORTS, CONTRACTS, CRIM. LAW, EVIDENCE, CONST. LAW, REAL PROP."]
        instruction_bank_subject_generation = [
            "Generate a bar exam multiple choice question, along with an explanation and answer, for the following subject: ",
            "Generate an MBE MC question for "]
        for idx, row in df.iterrows():
            # source_year = row["Source"].split("MBE-")[1].split("-")[0]
            if isinstance(row['Prompt'], str) and row['Prompt'].strip() != "" and row['Prompt'].strip() != "nan":
                question = row['Prompt']
            else:
                question = ""
            question += f" {row['Question']}"
            question = question.strip()
            choices = [row["Choice A"], row["Choice B"], row["Choice C"], row["Choice D"]]
            answer = row["Answer"]
            subject = row["Subject"]
            positive_passage = row["Positive Passage"]
            datapoint = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                lookup = ["A", "B", "C", "D"]
                datapoint += f"{lookup[i]}. {choice}\n"
            data_no_answer = datapoint
            datapoint += f"Explanation: {positive_passage}\nAnswer: {answer}"
            datapoint_with_answer = datapoint
            # if source_year.strip() != "" and int(source_year) > 1950 and int(source_year) < 2023:
            #     source_year_string = f" Consider only the law and relevant cases before {source_year}."
            # else:
            #     source_year_string = ""
            final_datapoint = self.random.choice(instructions_examples) + "\n\n" + datapoint
            yield self.build_data_point(prompt_language, "en", final_datapoint, task_type, jurisdiction)

            if isinstance(subject, str) and subject.strip() != "":
                datapoint = f"{self.random.choice(instruction_bank_subject)}\n\n{data_no_answer}\nSubject: {subject}"

                datapoint = self.random.choice(instructions_examples) + "\n\n" + datapoint
                yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)

                datapoint = self.random.choice(
                    instruction_bank_subject_generation) + subject + ".\n\n" + datapoint_with_answer
                yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)
