from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class ProfessionalLaw(AbstractDataset):
    def __init__(self):
        # TODO do we have an url here?
        super().__init__("ProfessionalLaw", "auxiliary_train_hendrycks_test")

    def get_data(self):
        # The first 1200 are extra bar exam questions, not sure if we want to keep these in
        instructions_examples = ["Generate some Multistate Bar Exam questions according to U.S. law.",
                                 "Answer these legal questions. Use American Law. A few examples are provided first to give the answer format.",
                                 "Answer these U.S. Multistate Bar Exam questions. A few examples are provided first to give the answer format.",
                                 "Pick the most correct option considering U.S. Law."]
        instructions_zero_shot = ["Answer these legal questions. Use American Law. Provide the choice as \"Answer:\"",
                                  "Answer these U.S. Multistate Bar Exam questions. Provide the choice as \"Answer:\"",
                                  "Pick the most correct option considering U.S. Law. Output the choice as \"Answer:\""]
        df = load_dataset("hendrycks_test", "professional_law", split="auxiliary_train").select(range(1200))
        task_type = TASK_TYPE.MULTIPLE_CHOICE
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        def shuffle_choices(choices: List[str], answer: int):
            x = list(enumerate(choices))
            self.random.shuffle(x)
            indices, choices = zip(*x)
            answer = indices.index(answer)
            return choices, answer

        for i, (this_question, this_choices, this_answer) in tqdm(enumerate(zip(
                df["question"], df["choices"], df["answer"]
        )), total=len(df)):
            prompt_samples = df.select(self.random.sample(list(range(0, i)) + list(range(i + 1, len(df))), 3))
            prompt = ""
            for j, (prompt_question, prompt_choices, prompt_answer) in enumerate(zip(
                    prompt_samples["question"], prompt_samples["choices"], prompt_samples["answer"]
            )):
                prompt += f"Question: {prompt_question}\n"
                lookup = ["(a)", "(b)", "(c)", "(d)"]
                prompt_choices, prompt_answer = shuffle_choices(prompt_choices, prompt_answer)
                for i, choice in enumerate(prompt_choices):
                    prompt += f"{lookup[i]} {choice}\n"
                prompt += (
                    f"The Final Answer: {lookup[prompt_answer]}\n\n"
                )
                prompt += "###\n\n"

            cur_question = prompt
            cur_question += f"Question: {this_question}\n"
            for i, choice in enumerate(this_choices):
                lookup = ["(a)", "(b)", "(c)", "(d)"]
                cur_question += f"{lookup[i]} {choice}\n"

            cur_question += (
                f"The Final Answer: {lookup[this_answer]}"
            )
            datapoint = cur_question

            final_datapoint = self.random.choice(instructions_examples) + "\n\n" + datapoint
            yield self.build_data_point(prompt_language, "en", final_datapoint, task_type, jurisdiction)

            datapoint_zero_shot = datapoint.replace("The Final Answer: ", "Answer: ").split("###")[-1].strip()
            final_datapoint_zero_shot = self.random.choice(instructions_zero_shot) + "\n\n" + datapoint_zero_shot
            yield self.build_data_point(prompt_language, "en", final_datapoint_zero_shot, task_type, jurisdiction)
