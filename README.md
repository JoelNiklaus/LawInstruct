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

## Tasks still to add:

- Contract extraction dataset (http://nlp.cs.aueb.gr/software_and_datasets/CONTRACTS_ICAIL2017/index.html
  , http://nlp.cs.aueb.gr/pubs/icail2017.pdf)
- US Caselaw Segmentation (https://github.com/jsavelka/us-dec-func-iss-sgm/blob/master/trade_secret_cases.json
  , http://ebooks.iospress.nl/volumearticle/50840)
- Cookie Policy Summarization (https://github.com/senjed/Summarization-of-Privacy-Policies
  , http://ceur-ws.org/Vol-2645/paper3.pdf)
- BVA Summarization (https://github.com/luimagroup/bva-summarization, https://dl.acm.org/doi/10.1145/3322640.3326728)
- Australian Case Citation Summarization (https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports)
- MAUD (https://github.com/TheAtticusProject/maud, https://huggingface.co/datasets/theatticusproject/maud)
- LegalCaseReports Summ (https://archive.ics.uci.edu/ml/machine-learning-databases/00239
  , https://aclanthology.org/W12-0515.pdf) ==> no re-destribution allowed ==> upload to private hf repo
- EurLexSum (https://huggingface.co/datasets/dennlinger/eur-lex-sum) ==> very long texts and summaries

Arya:

- LegalLinking (https://github.com/mayhewsw/legal-linking)
- Privacy Policies Summarization (https://github.com/senjed/Summarization-of-Privacy-Policies
- E-NER (https://github.com/terenceau2/E-NER-Dataset)
- GerDALIR (https://github.com/lavis-nlp/GerDaLIR)
- Dutch Legal Summarization (https://github.com/prijsdf/dutch-legal-summarization)
- Covid Law Matching (https://github.com/DFKI-NLP/covid19-law-matching)
- BUILD (https://legal-nlp-ekstep.github.io/Competitions/Rhetorical-Role/)
- CASS (https://github.com/euranova/CASS-dataset)
- ECHR Argument Mining (http://www.di.uevora.pt/~pq/echr/)
- Greek NER (https://github.com/nmpartzio/elNER)
- Indian NER (https://arxiv.org/pdf/2211.03442.pdf
  , https://github.com/Legal-NLP-EkStep/legal_NER/tree/main/representative_judgments_sample)
- LawngNLI (https://arxiv.org/pdf/2212.03222.pdf)
- Privacy Policies (https://usableprivacy.org/data) (excluding OPP-115 Corpus: already present in natural instructions)
- MakeThisYourLastTime (https://www.makethisyourlasttime.com/essay-bank/)

## Tasks to be reconsidered later

- LegalSum (https://github.com/sebimo/LegalSum) ==> complicated to read because of norms and would require large
  preprocessing. Additionally, contains very long sequences
- Indian/Australian Summarization (https://github.com/manavkapadnis/LegalEvaluation_LREC2022) ==> too long and for
  australian data, annotation done automatically
- BVACItationPrediction (https://github.com/TUMLegalTech/bva-citation-prediction) ==> no dataset downloadable directly
- BSARD (https://github.com/maastrichtlawtech/bsard) ==> legal articles are not available directly
- Cornell eRulemaking Corpus (https://facultystaff.richmond.edu/~jpark/data/jpark_lrec18.zip
  , https://facultystaff.richmond.edu/~jpark/papers/jpark_lrec18.pdf) ==> the full text of the comments is not available

## TODOs

- put local data on huggingface hub (find them if they use the raw_data folder)
- add urls to the source in the init call (MBE, civipro, mc_ecams, professional_law, sara_prolog)
- add more datasets from the list above
- add additional datasets (Arya)
- refactor code, so that all the instruction banks live in a json file that we can easily translate in the other languages
- translate instruction banks (from json file) into the 24 EU languages (Joel)
- use the same instruction banks for the same tasks if applicable
- test the dataset generation thoroughly
- run the script on a big machine to generate the datasets and upload to lawinstruct organisation on huggingface hub
- add more examples to the instruction banks

## Maybe later 
- translate some answers into the 24 EU languages ==> save instructions and answers into different columns 
- do not use xP3 and natural instructions but only code and legal instructions becuase of figure
  4: https://arxiv.org/pdf/2210.11416v5.pdf

## Done

- add initial datasets (Peter)
- code refactoring (Joel)
- add additional datasets (Joel)
- search for additional datasets (Joel)