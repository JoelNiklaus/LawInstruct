from datasets import load_dataset, get_dataset_config_names
import json

num_samples = 100
max_seq_len = 1024  # TODO test how much we can train LLaMA for on 4 80GB A100 GPUs
filename = f"law_instruction_data_{max_seq_len}.json"
use_fast_way = True  # there is a cleaner way which probably takes longer

dataset_name = "lawinstruct/lawinstruct"
configs = get_dataset_config_names(dataset_name)
print(configs)

instruction_data = []


def should_be_sampled(text):
    return text and len(text.split()) < max_seq_len


non_legal_configs = ['NaturalInstructionsOther', 'XP3MT']
faulty_configs = ['IndianTextSegmentation', 'Ell18Dataset', 'Ell4Dataset', 'EdgarNER']
configs = [config for config in configs
           if config not in non_legal_configs and config not in faulty_configs and config != 'all']

for config in configs:
    print(f"Loading {dataset_name}:{config}...")
    dataset = load_dataset(dataset_name, config, split="train", streaming=True)

    print(f"Filtering out examples with more than {max_seq_len} tokens and sampling {num_samples} examples...")
    if use_fast_way:
        num_samples_taken = 0
        for example in dataset:
            if should_be_sampled(example['text']):
                instruction_data.append(example)
                num_samples_taken += 1
            if num_samples_taken >= num_samples:
                break
    else:
        # this slows it down considerably for large datasets,
        # but could be more easily parallelized when using non-streaming datasets
        dataset = dataset.filter(lambda example: should_be_sampled(example['text']))
        dataset = dataset.shuffle(seed=42)
        instruction_data.extend(list(dataset.take(num_samples)))  # sample 100 examples

print(f"Writing {len(instruction_data)} examples to {filename}...")
with open(filename, "w") as file:
    json.dump(instruction_data, file, indent=4)
