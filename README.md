# LawInstruct

This repository has code used to generate legal instruction datasets

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

## Setup

Install the requirements from `requirements.txt`. Make sure to have python 3.10 or higher.
Make sure you have git and git-lfs installed.

On the ubelix slurm system, load the module with `module load git-lfs/2.4.2`
Run `git lfs install` to install git-lfs.

Clone the lawinstruct_raw repository locally:

```bash
git clone https://huggingface.co/datasets/lawinstruct/lawinstruct_raw
```

Clone the natural instructions data there too

```bash
git clone https://github.com/allenai/natural-instructions lawinstruct_raw/raw_data/ni_instructions_data
```

## TODOs

- run the script on a big machine to generate the datasets and upload to lawinstruct organisation on huggingface hub (
  Joel)
- use the same instruction banks for the same tasks if applicable (Lucia)
- add more examples to the instruction banks and diversify them by looking at FLAN and Natural Instructions (Lucia)
- make sure the jurisdiction is always in the instruction
- refactor code, so that we can allow for more finegrained instruction control (Arya)
- refactor code, so that all the instruction banks live in a json file that we can easily paraphrase and translate in
  the other languages (Arya)
- translate instruction banks (from json file) into the 24 EU languages (Joel)
- add retrieval datasets (see here for how to structure
  prompts: https://crfm-helm.readthedocs.io/en/latest/scenarios/#helm.benchmark.scenarios.msmarco_scenario)

## Maybe later

- put local data on huggingface hub (find them if they use the raw_data folder)
- translate some answers into the 24 EU languages ==> save instructions and answers into different columns
- do not use xP3 and natural instructions but only code and legal instructions because of figure
  4: https://arxiv.org/pdf/2210.11416v5.pdf
- add CoT data (https://github.com/jasonwei20/flan-2/blob/main/mmlu-cot.json) ==> this is only for MMMLU (which we leave
  out)

## Done

- add initial datasets (Peter)
- code refactoring (Joel)
- add additional datasets (Joel)
- search for additional datasets (Joel)
- add additional datasets (Arya)
- add more datasets like coliee (Lucia)
- replace ANSWER_GENERATION with QUESTION_ANSWERING (Joel)
- add urls to the source in the init call (MBE, civipro, mc_ecams, professional_law, sara_prolog)
- test the dataset generation thoroughly (Joel)

## Datasets still to add:

Arya:

- [-] ~~LawngNLI (https://arxiv.org/pdf/2212.03222.pdf, https://github.com/wbrun0/LawngNLI)~~  
  *24 GB unfiltered; I don't have space locally. -am*
- ~~[ ] ECHR Argument Mining (http://www.di.uevora.pt/~pq/echr/)~~  
  *This is an argument mining dataset.*

## Datasets possibly to be reconsidered later

Here we hit an obstacle

IR Datasets:

- GerDALIR (https://github.com/lavis-nlp/GerDaLIR) ==> This is an IR dataset.
- Covid Law Matching (https://github.com/DFKI-NLP/covid19-law-matching) ==> This is an IR dataset.
- BSARD (https://github.com/maastrichtlawtech/bsard) ==> This is an IR dataset.

Summarization Datasets:

- Dutch Legal Summarization (https://github.com/prijsdf/dutch-legal-summarization) ==> Requires multiple requests per
  document to retrieve; documentation in Dutch; no actual summarization targets.
- LegalSum (https://github.com/sebimo/LegalSum) ==> complicated to read because of norms and would require large
  preprocessing. Additionally, contains very long sequences
- Indian/Australian Summarization (https://github.com/manavkapadnis/LegalEvaluation_LREC2022) ==> too long and for
  australian data, annotation done automatically
- Cookie Policy Summarization (https://github.com/senjed/Summarization-of-Privacy-Policies
  , http://ceur-ws.org/Vol-2645/paper3.pdf) ==> automatic annotation, no summarization data available
- BVA Summarization (https://github.com/luimagroup/bva-summarization, https://dl.acm.org/doi/10.1145/3322640.3326728)
  ==> repo very badly documented, it is not clear which dataset to use
- LegalCaseReports Summ (https://archive.ics.uci.edu/ml/machine-learning-databases/00239
  , https://aclanthology.org/W12-0515.pdf) ==> no re-destribution allowed, thus upload to raw_data. (summaries not
  clear)

Other Datasets:

- BVACItationPrediction (https://github.com/TUMLegalTech/bva-citation-prediction) ==> no dataset downloadable directly
- Cornell eRulemaking Corpus (https://facultystaff.richmond.edu/~jpark/data/jpark_lrec18.zip
  , https://facultystaff.richmond.edu/~jpark/papers/jpark_lrec18.pdf) ==> the full text of the comments is not available
- US Caselaw Segmentation (https://github.com/jsavelka/us-dec-func-iss-sgm/blob/master/trade_secret_cases.json
  , http://ebooks.iospress.nl/volumearticle/50840) ==> sentence boundary detection is probably not the most useful task
- Contract extraction dataset (http://nlp.cs.aueb.gr/software_and_datasets/CONTRACTS_ICAIL2017/index.html
  , http://nlp.cs.aueb.gr/pubs/icail2017.pdf) ==> looks like a complicated dataset requiring preprocessing
- CASS (https://github.com/euranova/CASS-dataset) ==> Couldn't download - `wget` failed.
- LegalLinking (https://github.com/mayhewsw/legal-linking) ==> Could not recreate necessary Python environment.
- Privacy Policies (https://usableprivacy.org/data) (excluding OPP-115 Corpus: already present in natural instructions)
- MakeThisYourLastTime (https://www.makethisyourlasttime.com/essay-bank/) ==> Requires scraping several PDFs; format not
  standardized.
