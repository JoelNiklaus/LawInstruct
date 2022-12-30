import os
from datasets import load_dataset
from tqdm import tqdm

from utils import write_json_line, get_output_file_name, MAX_FILE_SIZE

try:
    import lzma as xz
except ImportError:
    import pylzma as xz

output_file_idx = 0
category = "xp3"
train_f = xz.open(get_output_file_name(category, output_file_idx), "wt")

# Include only code and datasets that we have legal data for
# Maybe also add 'zh', 'vi', because we have legal instruction datasets there
_LANG = ['en', 'es', 'fr', 'pt', 'code']
# TODO maybe treat code as a separate category so we can filter easily

# prompts translated into other languages
# rather use this one instead of xP3all because we compile multi_eurlex ourselves and xP3all only has two datasets more.
# We already have enough instruction datasets and rather prioritize using more languages in the prompts
for lang in _LANG:
    print("############################")
    print(f"########## bigscience/xP3mt {lang} ###########")
    print("############################")
    df = load_dataset("bigscience/xP3mt", lang)
    for example in tqdm(df["train"]):
        datapoint = example["inputs"] + " " + example["targets"]
        if os.path.getsize(get_output_file_name(category, output_file_idx)) > MAX_FILE_SIZE:
            train_f.close()
            output_file_idx += 1
            train_f = xz.open(get_output_file_name(category, output_file_idx))
        write_json_line(train_f, datapoint, lang, "https://huggingface.co/datasets/bigscience/xP3all")
train_f.close()
