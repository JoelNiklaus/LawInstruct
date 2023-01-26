from datasets import load_dataset
from tqdm import tqdm

from abstract_dataset import TASK_TYPE, JURISDICTION
from instruction_datasets.abstract_natural_instructions import \
    AbstractNaturalInstructions
from instruction_datasets.abstract_natural_instructions import get_lang_codes


class NaturalInstructionsOther(AbstractNaturalInstructions):

    def __init__(self):
        super().__init__("NaturalInstructionsOther", "https://github.com/allenai/natural-instructions")

    def get_data(self):
        raw_datasets = load_dataset('./raw_data/Tk-Instruct/src/ni_dataset.py', data_dir="raw_data/ni_task_configs",
                                    task_dir="./raw_data/ni_instructions_data/tasks", split="train")

        if self.filter_out_mmmlu:
            raw_datasets = raw_datasets.filter(lambda x: "mmmlu" not in x["Name"])
        prompt_language = "en"

        other_datasets = raw_datasets.filter(lambda x: x["Name"] not in self.legal_tasks.keys())

        for example in tqdm(other_datasets):
            lang_codes = get_lang_codes(example["Input_language"])
            task_type = TASK_TYPE.UNKNOWN
            jurisdiction = JURISDICTION.N_A

            for collator in self.collators:
                encoded_example = collator([example])
                # TODO additionally save prompt and label separately for the legal datasets so we can do machine translation only on prompt
                text = encoded_example["inputs"][0] + " " + encoded_example["labels"][0].strip()
                yield self.build_data_point(prompt_language, lang_codes, text, task_type, jurisdiction)
