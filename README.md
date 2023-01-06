# LawInstruct

This repository is a collection of legal instruction datasets

## How to add a new dataset

1. Take the raw data and upload it to the huggingface hub (in a private repo if the data is not permissively licensed)
2. Add a class to the folder `instruction_datasets` that inherits from `AbstractDataset` and implements the abstract
   method `get_data`. The `get_data` method should yield datapoints with the following fields:
    - "prompt_language": the language of the prompt
    - "answer_language": the language of the answer
    - "text": the prompt combined with the answer
    - "task_type": the type of task (e.g. "summarization")
    - "jurisdiction": the jurisdiction of the example (e.g. "US")
    - "subset": the subset of the dataset (e.g. "swiss_judgment_prediction" for "lextreme")
3. Add the dataset to the list in `build_instruction_datasets.py` a run the script to generate the dataset.
