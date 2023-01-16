from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class ShortAnswerFeedback(AbstractDataset):
    def __init__(self):
        super().__init__("ShortAnswerFeedback", "https://huggingface.co/datasets/JohnnyBoy00/saf_legal_domain_german")

    def get_data(self):
        df = load_dataset("JohnnyBoy00/saf_legal_domain_german")
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.GERMANY
        prompt_language = "en"

        instruction_bank_openqa = [
            "Consider this question in the context of German law. Provide the correct reference answer.",
            "Answer the question about German law. Make sure it is correct."]
        instruction_bank_feedback = [
            "Here is a question and answer pair related to German Law. Considering the student provided answer, provide detailed feedback and then provide a score of 1 for correct, 0.5 for partially correct, and 0 for incorrect.",
            "Consider the answer to the question, is it correct? Provide feedback and then give a score from 0 to 1.",
            "Consider the student's answer to the question. Rate it and provide feedback."]
        instruction_error_class = [
            "Here is a question and answer pair related to German Law. Considering the student provided answer, provide detailed feedback and then provide a score of 1 for correct, 0.5 for partially correct, and 0 for incorrect.",
            "Consider the answer to the question, is it correct? Provide feedback and then give a score from 0 to 1. Note the error class.",
            "Consider the student's answer to the question. Rate it and provide feedback. Note the type of error."]

        for example in df["train"]:
            text = f"{self.random.choice(instruction_bank_openqa)}\n\nQ: {example['question']}\nA: {example['reference_answer']}"
            yield self.build_data_point(prompt_language, "de", text, task_type, jurisdiction)

            text = f"{self.random.choice(instruction_bank_feedback)}\n\nQ: {example['question']}\nA: {example['provided_answer']}\nFeedback: {example['verification_feedback']}\nScore: {example['score']}"
            yield self.build_data_point(prompt_language, "de", text, task_type, jurisdiction)

            text = f"{self.random.choice(instruction_error_class)}\n\nQ: {example['question']}\nA: {example['provided_answer']}\nFeedback: {example['verification_feedback']}\nScore: {example['score']}\nError Type: {example['error_class']}"
            yield self.build_data_point(prompt_language, "de", text, task_type, jurisdiction)