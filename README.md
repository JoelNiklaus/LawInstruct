# LawInstruct

This repository has code used to generate legal instruction datasets.

## How to add a new dataset

1. If there is no public Hugging Face repo for the dataset, take the raw data and upload it to the Hugging Face hub (in a private repo if the data is not permissively licensed)
2. Add a class to the folder `instruction_datasets` that inherits from `AbstractDataset` and implements the abstract
   method `get_data`. The `get_data` method should yield datapoints with the following fields:
    - "instruction_language": the language of the instruction
    - "prompt_language": the language of the prompt
    - "answer_language": the language of the answer
    - "instruction": the instruction telling the model what to do
    - "prompt": the prompt input to the model
    - "answer": the answer providing the solution
    - "task_type": the type of task (e.g. "summarization")
    - "jurisdiction": the jurisdiction of the example (e.g. "US")
    - "subset": the subset of the dataset (e.g. "swiss_judgment_prediction" for "lextreme")
3. Write one to 10 seed instructions to the en.json file for the new class 
4. Add the dataset to the list in `lawinstruct_datasets.py`
5. To generate the dataset run the command 
   ```bash 
   python build_instruction_datasets.py --datasets ExampleDatasetName1 ExampleDatasetName2 --build_from_scratch
   ```

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

The en.json file was created by writing one to 5 seed instructions. Using GPT4, we generated paraphrases for each task.
We used the following prompt: "Below is a list of instructions for a large language model. Expand this json to 10
paraphrases. Provide json as output. Keep the provided examples."

## Possible improvements

- make huggingface dataset loading script better: enable dynamic loading of instructions in differing numbers of
  paraphrases and languages

## Maybe later

- add retrieval datasets (see here for how to structure
  prompts: https://crfm-helm.readthedocs.io/en/latest/scenarios/#helm.benchmark.scenarios.msmarco_scenario) ==> average
  prompt is very long, so we could probably only use a small part of the data
- frame casehold as a generation task: let the model generate the correct holding statement
- put local data on huggingface hub (find them if they use the raw_data folder)
- translate some answers into the 24 EU languages ==> save instructions and answers into different columns
- do not use xP3 and natural instructions but only code and legal instructions because of figure
  4: https://arxiv.org/pdf/2210.11416v5.pdf
- add CoT data (https://github.com/jasonwei20/flan-2/blob/main/mmlu-cot.json) ==> this is only for MMMLU (which we leave
  out)

## Datasets possibly to be reconsidered later

### Datasets to be added next
- Long-form Legal Question Answering (https://huggingface.co/datasets/maastrichtlawtech/lleqa)
  - ask for access 
  - add the question answering task similar to other qa tasks
  - try to think of other tasks that could be added like asking for regions or topics
- Legal Lens (https://huggingface.co/collections/joelniklaus/legallens-datasets-6601a17a92e663923265c845)
  - add the NER task similar to other ner tasks and the nli task similar to other nli tasks
- Swiss Citation Extraction (https://huggingface.co/datasets/rcds/swiss_citation_extraction/viewer/original/test?row=0)
  - combine the words again into a standard string but use the indices to edit it
  - two subsets:
    - citation extraction:
      - input: text with citations
      - output: list of citations (as a markdown list (- before each item and separated with new line))
    - citation prediction:
      - input: text with the citations masked with a placeholder (e.g., <citation1>, <citation2>, etc.)
      - output: - <citation1>: the citation
                - <citation2>: the citation
                - ...
- MultiLegalNeg (https://huggingface.co/datasets/rcds/MultiLegalNeg)
    - combine words similar to Swiss Citation Extraction
    - more information here: https://arxiv.org/pdf/2309.08695
    - two subsets:
      - negation cue detection: goal is for the language model to repeat the negation cue word
      - negation scope detection: goal is for the language model to repeat the negation scope given the negation cue
- Keyphrase Generation (https://huggingface.co/datasets/NCube/europa)

### Datasets where we hit an obstacle

#### IR Datasets:

- CLERC (https://huggingface.co/datasets/jhu-clsp/CLERC)
- GerDALIR (https://github.com/lavis-nlp/GerDaLIR)
- Covid Law Matching (https://github.com/DFKI-NLP/covid19-law-matching)
- BSARD (https://github.com/maastrichtlawtech/bsard)
- SwissIR (https://huggingface.co/datasets/rcds/doc2doc)

#### Summarization Datasets:

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

#### Other Datasets:

- BVACItationPrediction (https://github.com/TUMLegalTech/bva-citation-prediction) ==> no dataset downloadable directly
- Cornell eRulemaking Corpus (https://facultystaff.richmond.edu/~jpark/data/jpark_lrec18.zip
  , https://facultystaff.richmond.edu/~jpark/papers/jpark_lrec18.pdf) ==> the full text of the comments is not available
- US Caselaw Segmentation (https://github.com/jsavelka/us-dec-func-iss-sgm/blob/master/trade_secret_cases.json
  , http://ebooks.iospress.nl/volumearticle/50840) ==> sentence boundary detection is probably not the most useful task
- MultiLegalSBD (https://huggingface.co/datasets/rcds/MultiLegalSBD) ==> sentence boundary detection is probably not the
  most useful task
- Contract extraction dataset (http://nlp.cs.aueb.gr/software_and_datasets/CONTRACTS_ICAIL2017/index.html
  , http://nlp.cs.aueb.gr/pubs/icail2017.pdf) ==> looks like a complicated dataset requiring preprocessing
- CASS (https://github.com/euranova/CASS-dataset) ==> Couldn't download - `wget` failed.
- LegalLinking (https://github.com/mayhewsw/legal-linking) ==> Could not recreate necessary Python environment.
- Privacy Policies (https://usableprivacy.org/data) (excluding OPP-115 Corpus: already present in natural instructions)
- MakeThisYourLastTime (https://www.makethisyourlasttime.com/essay-bank/) ==> Requires scraping several PDFs; format not
  standardized.
- ECHR Argument Mining (http://www.di.uevora.pt/~pq/echr/) ==> This is an argument mining dataset.

## Troublehooting

Make sure to only yield from the same subset in the `get_data()` method. Otherwise, it will only write one example to
the file and close it again.

## References

Please cite the following preprint:

```
@misc{niklaus2024flawnt5,
      title={FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning}, 
      author={Joel Niklaus and Lucia Zheng and Arya D. McCarthy and Christopher Hahn and Brian M. Rosen and Peter Henderson and Daniel E. Ho and Garrett Honke and Percy Liang and Christopher Manning},
      year={2024},
      eprint={2404.02127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
