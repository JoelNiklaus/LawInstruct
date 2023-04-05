import json

from datasets import get_dataset_config_names
from datasets import load_dataset


use_fast_way = True  # there is a cleaner way which probably takes longer
dataset_name = "lawinstruct/lawinstruct"
configs = get_dataset_config_names(dataset_name)
print(configs)

non_legal_configs = ['NaturalInstructionsOther', 'XP3MT']
faulty_configs = ['IndianTextSegmentation', 'Ell18Dataset', 'Ell4Dataset', 'EdgarNER', 'SwissJudgmentPrediction']
configs = [config for config in configs
           if config not in non_legal_configs and config not in faulty_configs and config != 'all']


def generate_instruction_data(dataset_name, configs, max_seq_len=512, num_samples=500):
    def should_be_sampled(text):
        return text and len(text.split()) < max_seq_len

    instruction_data = []
    filename = f"law_instruction_data_len:{max_seq_len}_samples:{num_samples}.json"
    for config in configs:
        print(f"Loading {dataset_name}:{config}...")
        dataset = load_dataset(dataset_name, config, split="train", streaming=True)

        print(f"Filtering out examples with more than {max_seq_len} tokens and sampling {num_samples} examples...")
        if use_fast_way:
            num_samples_taken = 0
            for example in dataset:
                if should_be_sampled(example['text']):
                    instruction_data.append({"instruction": example["text"], "input": "", "output": ""})
                    num_samples_taken += 1
                if num_samples_taken >= num_samples:
                    break
        else:
            # this slows it down considerably for large datasets,
            # but could be more easily parallelized when using non-streaming datasets
            dataset = dataset.filter(lambda example: should_be_sampled(example['text']))
            dataset = dataset.shuffle(seed=42)
            examples_to_add = [{"instruction": example["text"], "input": "", "output": ""}
                               for example in dataset.take(num_samples)]
            instruction_data.extend(examples_to_add)  # sample 100 examples
    print(f"Writing {len(instruction_data)} examples to {filename}...")
    with open(filename, "w") as file:
        json.dump(instruction_data, file, indent=4)


if __name__ == '__main__':
    for max_seq_len in [512, 1024, 2048]:
        for num_samples in [100, 1000, 10000]:
            generate_instruction_data(dataset_name, configs, max_seq_len=max_seq_len, num_samples=num_samples)
