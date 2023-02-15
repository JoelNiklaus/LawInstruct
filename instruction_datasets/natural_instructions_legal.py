from datasets import load_dataset
from tqdm import tqdm

from instruction_datasets.abstract_natural_instructions import \
    AbstractNaturalInstructions
from instruction_datasets.abstract_natural_instructions import \
    get_first_lang_code


class NaturalInstructionsLegal(AbstractNaturalInstructions):

    def __init__(self):
        super().__init__("NaturalInstructionsLegal",
                         "https://github.com/allenai/natural-instructions")

    def get_data(self):
        raw_datasets = load_dataset(
            f'{self.raw_data_dir}/ni_dataset.py',
            data_dir=f"{self.raw_data_dir}/ni_task_configs",
            task_dir=f"{self.raw_data_dir}/ni_instructions_data/tasks",
            split="train")

        if self.filter_out_mmmlu:
            raw_datasets = raw_datasets.filter(
                lambda x: "mmmlu" not in x["Task"])
        prompt_language = "en"

        legal_datasets = raw_datasets.filter(
            lambda x: x["Task"] in self.legal_tasks.keys())

        for example in tqdm(legal_datasets):
            answer_language = get_first_lang_code(example["Input_language"])
            task_type = self.legal_tasks[example["Task"]]["task_type"]
            jurisdiction = self.legal_tasks[example["Task"]]["jurisdiction"]

            for collator in self.collators:
                encoded_example = collator([example])
                # TODO additionally save prompt and label separately for the legal datasets so we can do machine translation only on prompt
                text = encoded_example["inputs"][0] + " " + encoded_example[
                    "labels"][0].strip()
                yield self.build_data_point(prompt_language, answer_language,
                                            text, task_type, jurisdiction)
