import os
import re
from pprint import pprint


def scan_data_directory(directory):
    NUM_SHARDS = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl.xz'):
            base_name = os.path.splitext(filename)[0]
            base_name = re.sub(r'\.jsonl$', '', base_name)
            match = re.search(r'train-(\d+)', base_name)
            if match:
                number = int(match.group(1))
                key = re.sub(r'-train-\d+', '', base_name)
                NUM_SHARDS[key] = number + 1
    return NUM_SHARDS


if __name__ == '__main__':
    dataset = "english"
    num_shards = scan_data_directory(f'lawinstruct_{dataset}/data')
    pprint(num_shards)
