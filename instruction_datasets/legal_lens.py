from datasets import load_dataset

from abstract_dataset import AbstractDataset
from enums import Jurisdiction
from enums import TaskType
import instruction_manager
from greek_ner import NerTags

class LegalLensTags(NerTags):

    @property
    def _tags(self) -> list[str]:
        tags = ["O"]  # outside
        for position in ["B", "I"]:
            for type_ in [
                "LAW",
                "VIOLATION",
                "VIOLATED BY",
                "VIOLATED ON",
            ]:
                tags.append(f"{position}-{type_}")
        # Sanity checks
        assert "O" in tags
        assert "I-VIOLATED BY" in tags
        return tags
    
    @property
    def _delimiter(self) -> str:
        return ","

class LegalLens(AbstractDataset): 

    def __init__(self):
        super().__init__(
            "LegalLens",
            "https://huggingface.co/collections/joelniklaus/legallens-datasets-6601a17a92e663923265c845"
        )
        self._tags = LegalLensTags()


    def get_data(self, instructions: instruction_manager.InstructionManager): 
        df_nil = load_dataset("darrow-ai/LegalLensNLI", split = 'train')
        task_type = TaskType.NATURAL_LANGUAGE_INFERENCE
        jurisdiction = Jurisdiction.US
        instruction_language: str
        prompt_language = "en"
        answer_language = "en"
        for example in df_nil:
            subset = 'legal_lens_nli'
            instruction, instruction_language = instructions.sample(subset)
            promt = f"Case Context: {example['premise']}\n\n" \
                    f"Hypothesis: {example['hypothesis']}\n\n"\
                    f"Relevant Statute: {example['legal_act']}"
            answer = f"Classification: {example['label']}"
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction,
                                        promt, answer, task_type, 
                                        jurisdiction, subset)

        df_ner = load_dataset("darrow-ai/LegalLensNER", split = 'train')    
        task_type = TaskType.NAMED_ENTITY_RECOGNITION
        jurisdiction = Jurisdiction.US
        instruction_language = "en"
        answer_language = "ner"

        instruction_sentence = "Consider the following sentence for detecting legal violations."
        instruction_bank = [
            instruction_sentence + " " + self._tags.instruction
        ]

        for example in df_ner: 
            instruction = self.random.choice(instruction_bank)
            prompt, answer = self._tags.build_answer(df['tokens'], df['ner_tags'])
            yield self.build_data_point(instruction_language, prompt_language,
                                        answer_language, instruction, prompt,
                                        answer, task_type, jurisdiction)