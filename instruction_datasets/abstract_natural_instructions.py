from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE

import string
from transformers.data.data_collator import *

from iso639 import languages


@dataclass
class DataCollatorForNI:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool = False

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0,
                     "add_explanation": False},
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
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation

            task_input = ""
            # add the input first.
            # task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip()  # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                if add_explanation:
                    random.choice(["If you can, please add an explanation *before* you output your answer.",
                                   "Please output an explanation first and then come to your conclusion and create an output.",
                                   "Explain your answer first.",
                                   "Think step by step before outputting an answer.",
                                   "Explain yourself."])
                definition += "\n\n"

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx + 1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += random.choice([f" Explanation: {pos_example['explanation'].strip()}",
                                                      f" Let's think step by step. {pos_example['explanation'].strip()}"])
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                # if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                pos_examples.append(pos_example_str)
                # else:
                #     break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx + 1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += random.choice([f" Explanation: {neg_example['explanation'].strip()}",
                                                      f" Let's think step by step. {neg_example['explanation'].strip()}"])
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                # if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                neg_examples.append(neg_example_str)
                # else:
                #     break

            source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
            # tokenized_source = self.tokenizer(source)["input_ids"]
            # if len(tokenized_source) <= self.max_source_length:
            sources.append(source)
            # else:
            #     sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model,
                                              "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs


class AbstractNaturalInstructions(AbstractDataset):
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

    # searched by "Law", "Legal", "Jurisprudence": https://github.com/allenai/natural-instructions/tree/master/tasks
    legal_tasks = {
        'task268_casehold_legal_answer_generation': {"jurisdiction": JURISDICTION.US,
                                                     "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task274_overruling_legal_classification': {"jurisdiction": JURISDICTION.US,
                                                    "task_type": TASK_TYPE.TEXT_CLASSIFICATION},
        'task287_casehold_legal_incorrect_answer_generation': {"jurisdiction": JURISDICTION.US,
                                                               "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task597_cuad_answer_generation': {"jurisdiction": JURISDICTION.US, "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task598_cuad_answer_generation': {"jurisdiction": JURISDICTION.US, "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task599_cuad_question_generation': {"jurisdiction": JURISDICTION.US,
                                             "task_type": TASK_TYPE.QUESTION_GENERATION},
        'task683_online_privacy_policy_text_purpose_answer_generation': {"jurisdiction": JURISDICTION.UNKWNOWN,
                                                                         "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task684_online_privacy_policy_text_information_type_generation': {"jurisdiction": JURISDICTION.UNKOWN,
                                                                           "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task715_mmmlu_answer_generation_international_law': {"jurisdiction": JURISDICTION.INTERNATIONAL,
                                                              "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task716_mmmlu_answer_generation_jurisprudence': {"jurisdiction": JURISDICTION.US,
                                                          "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task729_mmmlu_answer_generation_professional_law': {"jurisdiction": JURISDICTION.US,
                                                             "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task743_eurlex_summarization': {"jurisdiction": JURISDICTION.EU, "task_type": TASK_TYPE.SUMMARIZATION},
        'task744_eurlex_classification': {"jurisdiction": JURISDICTION.EU, "task_type": TASK_TYPE.TEXT_CLASSIFICATION},
        'task1658_billsum_summarization': {"jurisdiction": JURISDICTION.US, "task_type": TASK_TYPE.SUMMARIZATION},
        'task1666_cail2018_answer_generation': {"jurisdiction": JURISDICTION.CHINA,
                                                "task_type": TASK_TYPE.ANSWER_GENERATION},
        'task1667_cail2018_answer_generation': {"jurisdiction": JURISDICTION.CHINA,
                                                "task_type": TASK_TYPE.ANSWER_GENERATION},
    }

    def __init__(self, name, source):
        super().__init__(name, source)
        self.collators = []
        for encoding in self.all_valid_encodings:
            self.collators.append(DataCollatorForNI(
                tokenizer=None,
                model=None,
                **encoding,
                text_only=True
            ))
        self.filter_out_mmmlu = True

    def get_lang_codes(self, langs):
        lang_codes = []
        for lang in langs:
            try:
                lang_code = languages.get(name=lang).alpha2
            except KeyError:
                lang_code = "unknown"
            lang_codes.append(lang_code)
            return lang_codes