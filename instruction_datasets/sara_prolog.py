import os

from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class SaraProlog(AbstractDataset):
    def __init__(self):
        super().__init__("SaraProlog", "https://arxiv.org/abs/2005.05257")

    def get_data(self):
        instruction_bank = ["Convert the following statute into prolog code.",
                            "Write a prolog program to convert the statute into code, denote it as \"Prolog Program:\"."]
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        json_files = [pos_json for pos_json in os.listdir(f"{self.raw_data_dir}/sara_statutes/source")]
        for json_file in json_files:
            with open(os.path.join(f"{self.raw_data_dir}/sara_statutes/source/", json_file), "r") as f_normal:
                with open(os.path.join(f"{self.raw_data_dir}/sara_statutes/prolog/", json_file) + ".pl", "r") as f_prolog:
                    datapoint = f"{self.random.choice(instruction_bank)}\n\nStatute:\n{f_normal.read()}\n\nProlog Program:\n\n{f_prolog.read()}"
                    yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)

        instruction_bank = ["Convert the following fact pattern into prolog code. Then answer the question.",
                            "Write a prolog program to mark all the facts, denote it as \"Prolog Program:\". Then answer the question, denote your answer as \"Answer\"."]
        json_files = [pos_json for pos_json in os.listdir(f"{self.raw_data_dir}/sara_cases/") if pos_json != "train"]
        with open(f"{self.raw_data_dir}/sara_cases/train", "r") as train_list_f:
            train_list = [x.strip() for x in train_list_f.readlines()]
        for json_file in json_files:
            with open(os.path.join(f"{self.raw_data_dir}/sara_cases/", json_file), "r") as f_normal:
                if json_file.split(".pl")[0] not in train_list:
                    print(f"Skipping {json_file}")
                text = f_normal.read()

                facts_and_question = text.split("% Facts")[0]
                program = text.split("% Facts")[1]

                if "Entailment" in facts_and_question:
                    answer = "True"
                elif "Contradiction" in facts_and_question:
                    answer = "False"
                else:
                    answer = facts_and_question.split("% Question")[1].split("?")[-1].strip()

                facts_and_question = facts_and_question.replace("Entailment", "Is this True or False?")
                facts_and_question = facts_and_question.replace("Contradiction", "Is this True or False?")
                facts_and_question = facts_and_question.replace("\n", " ")
                facts_and_question = facts_and_question.replace("% Text", "Facts:")
                facts_and_question = facts_and_question.replace("% Question", "\nQuestion:")
                facts_and_question = facts_and_question.replace("%", "").strip()

                datapoint = f"{self.random.choice(instruction_bank)}\n\n{facts_and_question}\n\nProlog Program:\n{program.strip()}\nAnswer: {answer}"
                yield self.build_data_point(prompt_language, "en", datapoint, task_type, jurisdiction)
