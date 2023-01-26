import json

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE


class CABarExamEssays(AbstractDataset):
    def __init__(self):
        super().__init__("CABarExamEssays", "https://www.calbar.ca.gov/Admissions/Examinations/California-Bar-Examination/Past-Exams")

    def get_data(self):
        # Scraped bar exam essays
        task_type = TASK_TYPE.QUESTION_ANSWERING
        jurisdiction = JURISDICTION.US
        prompt_language = "en"

        with open("raw_data/bar_exam_essays_ca.jsonl") as f:
            exams = [json.loads(x) for x in f.readlines()]
            for exam in exams:
                text = exam['text']
                yield self.build_data_point(prompt_language, "en", text, task_type, jurisdiction)
