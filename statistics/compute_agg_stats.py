from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import os

from tqdm import tqdm
import copy

def get_length(text):
    if not text:
        return 0
    return len(text.split())


def compute_dataset_stats():
    for config in configs:
        print(f"Started computing dataset specific stats for {config}")
        stats = copy.deepcopy(base_dict)

        dataset = load_dataset(dataset_name, config, split="train", streaming=True)
        for example in tqdm(dataset):
            stats["jurisdiction"].append(example["jurisdiction"])
            stats["task_type"].append(example["task_type"])

            stats["instruction_language"].append(example["instruction_language"])
            stats["prompt_language"].append(example["prompt_language"])
            stats["answer_language"].append(example["answer_language"])

            stats["instruction_length"].append(get_length(example["instruction"]))
            stats["prompt_length"].append(get_length(example["prompt"]))
            stats["answer_length"].append(get_length(example["answer"]))

        # convert stats to pandas dataframe and save to csv
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(f"{folder_name}/{config}.csv", index=False)
        print(f"Finished writing dataset specific stats for {config}")


def compute_aggregate_stats():
    # Aggregate stats
    agg = {"dataset": [], "subset": []}
    agg.update(copy.deepcopy(base_dict))
    print(f"Started aggregating stats for configs {configs}")
    for config in tqdm(configs):
        # load dataset statistics
        df = pd.read_csv(f"{folder_name}/{config}.csv")

        # add general information
        dataset, subset = config.split("-", 2)
        agg["dataset"].append(dataset)
        agg["subset"].append(subset)

        # add categorical information
        for categorical in categoricals:
            agg[categorical].append(df[categorical].value_counts().to_dict())

        # add numerical information
        for numerical in numericals:
            agg[numerical].append(df[numerical].describe().to_dict())
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(f"aggregate_stats.csv", index=False)
    return agg_df


if __name__ == '__main__':
    dataset_name = "lawinstruct/lawinstruct_multilingual"
    configs = get_dataset_config_names(dataset_name)
    configs = [config for config in configs if config != 'all']
    print(f"Computing stats for configs {configs}")

    # Collect dataset specific stats
    folder_name = "dataset_stats"
    os.makedirs(folder_name, exist_ok=True)

    categoricals = ["jurisdiction", "task_type",
                    "instruction_language", "prompt_language", "answer_language"]
    numericals = ["instruction_length", "prompt_length", "answer_length"]

    base_dict = {}
    base_dict.update({categorical: [] for categorical in categoricals})
    base_dict.update({numerical: [] for numerical in numericals})

    compute_dataset_stats()
    compute_aggregate_stats()
