import os
from datasets import load_dataset
from tqdm import tqdm

from utils import write_json_line, get_output_file_name

try:
    import lzma as xz
except ImportError:
    import pylzma as xz
from ni_collator import DataCollatorForNI

output_file_idx = 0
category = "natural_instructions"
train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")

print("############################")
print("########## natural instructions ###########")
print("############################")
raw_datasets = load_dataset('./raw_data/Tk-Instruct/src/ni_dataset.py', data_dir="raw_data/ni_task_configs",
                            task_dir="./raw_data/ni_instructions_data/tasks")

# tasks = set(x["train"]["Task"])
# block_list = ["mmlu"]
# for task in tasks:
#     for block in block_list:
#         if block in task:
#             continue
all_valid_encodings = [
    # instruction only
    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0,
     "add_explanation": False},
    # instruction + explanation
    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0,
     "add_explanation": True},
    # example only
    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0,
     "add_explanation": False},
    # instruction + pos examples
    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0,
     "add_explanation": False},
    # instruction + pos examples + neg examples
    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2,
     "add_explanation": False},
    # instruction + pos (w. explanation)
    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0,
     "add_explanation": True},
]
collators = []
for encoding in all_valid_encodings:
    collators.append(DataCollatorForNI(
        tokenizer=None,
        model=None,
        **encoding,
        text_only=True
    ))
for example in tqdm(raw_datasets["train"]):
    for collator in collators:
        encoded_example = collator([example])
        datapoint = encoded_example["inputs"][0] + " " + encoded_example["labels"][0].strip()
        if os.path.getsize(get_output_file_name(category, output_file_idx)) > 6.25e8:
            train_f.close()
            output_file_idx += 1
            train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")
        write_json_line(train_f, datapoint, example["Input_language"], example["URL"])
    # prompt = prompt[:-3].strip()
    # prompt_with_explanation = prompt_with_explanation[:-3].strip()
    # prompt_with_explanation_last = prompt_with_explanation_last[:-3].strip()
    # datapoint = f"{example['Definition']}\n\n{prompt}"
