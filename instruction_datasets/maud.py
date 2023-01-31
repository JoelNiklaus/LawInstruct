from datasets import load_dataset

from abstract_dataset import AbstractDataset, JURISDICTION, TASK_TYPE

INFO_STR = """
CATEGORY: General Information
    TEXT_TYPE: Type of Consideration
    	QUESTION 1: Type of Consideration-Answer
    		ANSWER 1: All Cash
    		ANSWER 2: All Stock
    		ANSWER 3: Mixed Cash/Stock
    		ANSWER 4: Mixed Cash/Stock: Election
CATEGORY: Conditions to Closing
    TEXT_TYPE: Accuracy of Target R&W Closing Condition
    	QUESTION 1: Accuracy of Target \"General\" R&W: Bringdown Timing Answer
    		ANSWER 1: At Closing Only
    		ANSWER 2: At Signing & At Closing
    	QUESTION 2: Accuracy of Target \"General\" R&W: Bringdown Standard Answer
    		ANSWER 1: All/The R&Ws accurate at MAE standard
    		ANSWER 2: All/The R&Ws accurate in all respects (repeating R&Ws)
    		ANSWER 3: Each R&W accurate at MAE standard
    		ANSWER 4: R&Ws accurate at another materiality standard (e.g., hybrid standard)
    	QUESTION 3: Accuracy of Target Capitalization R&W (outstanding shares): Bringdown Standard Answer
    		ANSWER 1: Accurate in all material respects
    		ANSWER 2: Accurate in all respects
    		ANSWER 3: Accurate in all respects with below-threshold carveout
    		ANSWER 4: Accurate in all respects with de minimis exception
    	QUESTION 4: Accuracy of Fundamental Target R&Ws: Bringdown Standard
    		ANSWER 1: Accurate at another materiality standard (e.g., hybrid standard)
    		ANSWER 2: Accurate in all material respects
    		ANSWER 3: Accurate in all respects
    	QUESTION 5: Accuracy of Fundamental Target R&Ws-Types of R&Ws[MULTILABEL]
    		ANSWER 1: Approval
    		ANSWER 2: Authority
    		ANSWER 3: Brokers' Fee
    		ANSWER 4: Capitalization-Other
    		ANSWER 5: Enforceability
    		ANSWER 6: No MAE
    		ANSWER 7: No-Conflict
    		ANSWER 8: Opinion of Financial Advisor
    		ANSWER 9: Organization
    		ANSWER 10: Other
    		ANSWER 11: Rights Agreement
    		ANSWER 12: Subsidiaries
    		ANSWER 13: Takeover Statutes
    		ANSWER 14: Tax
    	QUESTION 6: Materiality/MAE Scrape applies to[MULTILABEL]
    		ANSWER 1: Capitalization R&Ws
    		ANSWER 2: General R&Ws
    		ANSWER 3: Specified R&Ws only
    		ANSWER 4: fundamental/Special R&Ws
    TEXT_TYPE: Compliance with Covenant Closing Condition
    	QUESTION 1: Compliance with Target Covenant Closing Condition-Answer
    		ANSWER 1: All Covenants
    		ANSWER 2: Each Covenant
    		ANSWER 3: Hybrid/Other Standard
    TEXT_TYPE: Absence of Litigation Closing Condition
    	QUESTION 1: Absence of Litigation Closing Condition: Governmental v. Non-Governmental-Answer
    		ANSWER 1: Governmental litigation only
    		ANSWER 2: Non-Governmental & governmental litigation
    	QUESTION 2: Absence of Litigation Closing Condition: Pending v. Threatened v. Threatened in Writing-Answer
    		ANSWER 1: Pending
    		ANSWER 2: Pending or threatened (without \"writing\" requirement)
    		ANSWER 3: Pending or threatened in writing
CATEGORY: Material Adverse Effect
    TEXT_TYPE: MAE Definition
    	QUESTION 1: MAE definition includes adverse impact on Target's ability to consummate (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 2: \"Ability to consummate\" concept is subject to MAE carveouts
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 3: MAE definition includes reference to Target \"prospects\" (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 4: MAE Forward looking standard (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 5: FLS (MAE) applies to[MULTILABEL]
    		ANSWER 1: No
    		ANSWER 2: ability to consummate transaction
    		ANSWER 3: business and operation of Target
    	QUESTION 6: FLS (MAE) Standard-Answer
    		ANSWER 1: \"Could\" (reasonably) be expected to
    		ANSWER 2: \"Would\"
    		ANSWER 3: \"Would\" (reasonably) be expected to
    		ANSWER 4: No
    		ANSWER 5: Other forward-looking standard
    	QUESTION 7: MAE applies to Target and subsidiaries (MAE)-Answer
    		ANSWER 1: Applies to Target and subsidiaries \"taken as a whole\"
    		ANSWER 2: No
    	QUESTION 8: General political and/or social conditions (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 9: General political and/or social conditions:  subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 10: General economic and financial conditions (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 11: General economic and financial conditions: subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 12: Changes in Target's industry (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 13: Change in Target's industry: subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 14: Change in law (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 15: Change in law:  subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 16: Changes in GAAP or other accounting principles (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 17: Changes in GAAP or other accounting principles:  subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 18: Announcement, pendency or consummation of deal (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 19: A/P/C application to-Answer [MULTILABEL]
    		ANSWER 1: No
    		ANSWER 2: announcement
    		ANSWER 3: consummation
    		ANSWER 4: pendency
    	QUESTION 20: Failure to meet projections (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 21: Failure to meet projections: subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 22: Changes in market price/trading volume of Target's securities or credit rating (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 23: Target's securities or credit rating: subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 24: War, terrorism, natural disasters, \"acts of God\" or force majeure-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 25: W/N/A/F applies to-Answer [MULTILABEL]
    		ANSWER 1: \"act of God\"
    		ANSWER 2: Natural disaster
    		ANSWER 3: No
    		ANSWER 4: War or terrorism
    		ANSWER 5: force majeure
    	QUESTION 26: W/N/A/F subject to \"disproportionate impact\"-Answer[MULTILABEL]
    		ANSWER 1: \"act of God\"
    		ANSWER 2: Natural disaster
    		ANSWER 3: No
    		ANSWER 4: War or terrorism
    		ANSWER 5: force majeure
    	QUESTION 27: Pandemic or other public health event-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 28: Pandemic or other public health event: Specific reference to COVID-19
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 29: Pandemic or other public health event: specific reference to pandemic-related governmental responses or measures
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 30: Pandemic or other public health event:  subject to \"disproportionate impact\" modifier
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 31: Actions required under transaction agreement-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 32: Actions taken with consent or approval of Buyer-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 33: Target stockholder proceedings-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 34: Matters listed on disclosure schedules-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 35: Actions taken by Buyer-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 36: Action prohibited/omission required by the agreement-Answer
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 37: Relational language (MAE carveout)-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 38: Relational language (MAE carveout)-Answer (Dropdown) [MULTILABEL]
    		ANSWER 1: \"Arising from/out of\"
    		ANSWER 2: \"Attributable to\"
    		ANSWER 3: \"Relating to\"
    		ANSWER 4: \"Resulting from\"
    		ANSWER 5: No
    		ANSWER 6: Other
    		ANSWER 7: Relational language varies among carveouts
    	QUESTION 39: Relational language (MAE) applies to
    		ANSWER 1: All MAE carveouts
    		ANSWER 2: No
    		ANSWER 3: Some MAE carveouts
CATEGORY: Knowledge
    TEXT_TYPE: Knowledge Definition
    	QUESTION 1: Knowledge Definition-Answer
    		ANSWER 1: Actual knowledge
    		ANSWER 2: Constructive knowledge
    	QUESTION 2: Constructive Knowledge-Answer
    		ANSWER 1: Based on investigation or inquiry
    		ANSWER 2: Based on role
    	QUESTION 3: Knowledge Definition limited to one or more identified persons-Answer (Y/
    		ANSWER 1: No
    		ANSWER 2: Yes
CATEGORY: Operating and Efforts Covenant
    TEXT_TYPE: Ordinary course covenant
    	QUESTION 1: Buyer consent requirement (ordinary course)-Answer
    		ANSWER 1: Consent may not be unreasonably withheld, conditioned or delayed
    		ANSWER 2: Flat consent
    	QUESTION 2: Includes \"consistent with past practice\"
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 3: Ordinary course efforts standard-Answer
    		ANSWER 1: Commercially reasonable efforts
    		ANSWER 2: Flat covenant (no efforts standard)
    		ANSWER 3: Reasonable best efforts
    	QUESTION 4: Ordinary Course Covenant includes carve-out for Pandemic responses-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    TEXT_TYPE: Negative interim operating covenant
    	QUESTION 1: Buyer consent requirement (negative interim covenant)-Answer
    		ANSWER 1: Consent may not be unreasonably withheld, conditioned or delayed
    		ANSWER 2: Flat consent
    	QUESTION 2: Application of Buyer consent requirement (negative interim covenant)-Answer
    		ANSWER 1: Applies only to specified negative covenants
    		ANSWER 2: Applies to all negative covenants
    	QUESTION 3: Negative Interim Covenant includes carveout for pandemic responses-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    TEXT_TYPE: General Antitrust Efforts Standard
    	QUESTION 1: General Antitrust Efforts Standard-Answer
    		ANSWER 1: Commercially reasonable efforts
    		ANSWER 2: Flat standard
    		ANSWER 3: Reasonable best efforts
CATEGORY: Deal Protection and Related Provisions
    TEXT_TYPE: No-Shop
    	QUESTION 1: Liability for breaches of no-shop by Target Representatives (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 2: Liability standard for no-shop breach by Target Non-D&O Representatives
    		ANSWER 1: Reasonable standard
    		ANSWER 2: Strict liability
    TEXT_TYPE: Fiduciary exception:  Board determination (no-shop)
    	QUESTION 1: Fiduciary exception:  Board determination standard-Answer (no-shop)
    		ANSWER 1: \"Breach\" of fiduciary duties
    		ANSWER 2: \"Inconsistent\" with fiduciary duties
    		ANSWER 3: \"Reasonably likely/expected breach\" of fiduciary duties
    		ANSWER 4: \"Reasonably likely/expected to be inconsistent\" with fiduciary duties
    		ANSWER 5: \"Reasonably likely/expected violation\" of fiduciary duties
    		ANSWER 6: \"Required to comply\" with fiduciary duties
    		ANSWER 7: \"Violation\" of fiduciary duties
    		ANSWER 8: None
    		ANSWER 9: Other specified standard
    	QUESTION 2: Fiduciary exception: Board determination trigger (no shop)-Answer
    		ANSWER 1: Acquisition Proposal only
    		ANSWER 2: Superior Offer, or Acquisition Proposal reasonably likely/expected to result in a Superior Offer
    TEXT_TYPE: Fiduciary exception to COR covenant
    	QUESTION 1: COR permitted with board fiduciary determination only
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 2: COR standard (board determination only)-answer
    		ANSWER 1: \"Reasonably likely/expected breach\" of fiduciary duties
    		ANSWER 2: \"Reasonably likely/expected to be inconsistent\" with fiduciary duties
    		ANSWER 3: More likely than not violate fiduciary duties
    		ANSWER 4: Other specified standard
    	QUESTION 3: COR standard (superior offer)
    		ANSWER 1: \"Breach\" of fiduciary duties
    		ANSWER 2: \"Inconsistent\" with fiduciary duties
    		ANSWER 3: \"Reasonably likely/expected breach\" of fiduciary duties
    		ANSWER 4: \"Reasonably likely/expected to be inconsistent\" with fiduciary duties
    		ANSWER 5: \"Reasonably likely/expected violation\" of fiduciary duties
    		ANSWER 6: \"Required to comply\" with fiduciary duties
    		ANSWER 7: \"Violation\" of fiduciary duties
    		ANSWER 8: More likely than not violate fiduciary duties
    		ANSWER 9: None
    		ANSWER 10: Other specified standard
    	QUESTION 4: COR permitted in response to Intervening Event
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 5: COR standard (intervening event)
    		ANSWER 1: \"Breach\" of fiduciary duties
    		ANSWER 2: \"Inconsistent\" with fiduciary duties
    		ANSWER 3: \"Reasonably likely/expected breach\" of fiduciary duties
    		ANSWER 4: \"Reasonably likely/expected to be inconsistent\" with fiduciary duties
    		ANSWER 5: \"Reasonably likely/expected violation\" of fiduciary duties
    		ANSWER 6: \"Required to comply\" with fiduciary duties
    		ANSWER 7: \"Violation\" of fiduciary duties
    		ANSWER 8: More likely than not violate fiduciary duties
    		ANSWER 9: Other specified standard
    TEXT_TYPE: Agreement provides for matching rights in connection with COR
    	QUESTION 1: Initial matching rights period (COR)-Answer
    		ANSWER 1: 2 business days or less
    		ANSWER 2: 3 business days
    		ANSWER 3: 3 calendar days
    		ANSWER 4: 4 business days
    		ANSWER 5: 4 calendar days
    		ANSWER 6: 5 business days
    		ANSWER 7: Greater than 5 business days
    	QUESTION 2: Additional matching rights period for modifications (COR)-Answer
    		ANSWER 1: 2 business days or less
    		ANSWER 2: 3 business days
    		ANSWER 3: 3 days
    		ANSWER 4: 4 business days
    		ANSWER 5: 5 business days
    		ANSWER 6: > 5 business days
    		ANSWER 7: None
    	QUESTION 3: Number of additional matching rights periods for modifications (COR)
    		ANSWER 1: Continuous matching right
    		ANSWER 2: None
    TEXT_TYPE: Superior Offer Definition
    	QUESTION 1: Definition includes stock deals-Answer
    		ANSWER 1: \"All or substantially all\"
    		ANSWER 2: 50%
    		ANSWER 3: Greater than 50% but not \"all or substantially all\"
    		ANSWER 4: Less than 50%
    	QUESTION 2: Definition includes asset deals-Answer
    		ANSWER 1: \"All or substantially all\"
    		ANSWER 2: 50%
    		ANSWER 3: Greater than 50% but not \"all or substantially all\"
    		ANSWER 4: Less than 50%
    	QUESTION 3: \"Financial point of view\" is the sole consideration
    		ANSWER 1: No
    		ANSWER 2: Yes
    TEXT_TYPE: Intervening Event Definition
    	QUESTION 1: Definition contains knowledge requirement - answer
    		ANSWER 1: Known, but consequences unknown or not reasonably foreseeable, at signing
    		ANSWER 2: Known, but consequences unknown, at signing
    		ANSWER 3: Not known and not reasonably foreseeable at signing
    		ANSWER 4: Not known at signing
    	QUESTION 2: Knowledge persons include Target management (intervening event)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 3: Intervening Event - Required to Occur After Signing - answer
    		ANSWER 1: May occur or arise prior to signing
    		ANSWER 2: Must occur or arise after signing
    	QUESTION 4: Definition contains a materiality standard (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    TEXT_TYPE: FTR Triggers
    	QUESTION 1: FTR Triggers-Answer
    		ANSWER 1: Superior Offer
    		ANSWER 2: Superior Offer, Intervening Event
    TEXT_TYPE: Limitations on FTR Exercise
    	QUESTION 1: Limitations on FTR Exercise-Answer [MULTILABEL]
    		ANSWER 1: (Material) breach of other provisions of agreement
    		ANSWER 2: Any breach of no-shop
    		ANSWER 3: Breach of no-shop resulting in a Superior Offer
    		ANSWER 4: Material breach of no-shop
    		ANSWER 5: Material breach of no-shop resulting in a Superior Offer
    		ANSWER 6: Other
    TEXT_TYPE: Agreement provides for matching rights in connection with FTR
    	QUESTION 1: Initial matching rights period (FTR)-Answer
    		ANSWER 1: 2 business days or less
    		ANSWER 2: 3 business days
    		ANSWER 3: 3 calendar days
    		ANSWER 4: 4 business days
    		ANSWER 5: 4 calendar days
    		ANSWER 6: 5 business days
    		ANSWER 7: 5 calendar days
    		ANSWER 8: Greater than 5 business days
    	QUESTION 2: Additional matching rights period for modifications (FTR)-Answer
    		ANSWER 1: 2 business days or less
    		ANSWER 2: 3 business days
    		ANSWER 3: 3 days
    		ANSWER 4: 4 business days
    		ANSWER 5: 5 business days
    		ANSWER 6: >5 business days
    	QUESTION 3: Number of additional matching rights periods for modifications (FTR)
    		ANSWER 1: Continuous matching right
    		ANSWER 2: None
    TEXT_TYPE: Tail Period & Acquisition Proposal Details
    	QUESTION 1: Acquisition Proposal Timing-Answer [MULTILABEL]
    		ANSWER 1: Different Acquisition Proposal - Must be approved or not opposed (or another similar action) during Tail Period (no closing requirement)
    		ANSWER 2: Different Acquisition Proposal - Must be approved or not opposed (or another similar action) during Tail Period and transaction must close after Tail Period
    		ANSWER 3: Different Acquisition Proposal - Must sign during Tail Period (no closing requirement)
    		ANSWER 4: Different Acquisition Proposal - Must sign during Tail Period and close after Tail Period
    		ANSWER 5: Different Acquisition Proposal must close during Tail Period
    		ANSWER 6: Same Acquisition Proposal - Must be approved or not opposed (or another similar action) during Tail Period and transaction must close after Tail Period
    		ANSWER 7: Same Acquisition Proposal - Must be approved or not opposed to (or another similar action) during Tail Period (no closing requirement)
    		ANSWER 8: Same Acquisition Proposal - Must close during Tail Period
    		ANSWER 9: Same Acquisition Proposal - Must sign during Tail Period (no closing requirement)
    		ANSWER 10: Same Acquisition Proposal - Must sign during Tail Period and close after Tail Period
    	QUESTION 2: Acquisition Proposal required to be publicly disclosed-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 3: Acquisition Proposal required to be publicly disclosed-Answer [MULTILABEL]
    		ANSWER 1: No
    		ANSWER 2: “Publicly disclosed” requirement applies to Acquisition Proposal + Breach Trigger
    		ANSWER 3: “Publicly disclosed” requirement applies to Acquisition Proposal + No-Vote / MTC Failure Trigger
    		ANSWER 4: “Publicly disclosed” requirement applies to Acquisition Proposal + Outside Date Trigger
    	QUESTION 4: Acquisition Proposal required to be still pending-Answer (Y/N)
    		ANSWER 1: No
    		ANSWER 2: Yes
    	QUESTION 5: Tail Period Length-Answer
    		ANSWER 1: 12 months or longer
    		ANSWER 2: Other
    		ANSWER 3: within 12 months
    		ANSWER 4: within 6 months
    		ANSWER 5: within 9 months
    TEXT_TYPE: Breach of No Shop
    	QUESTION 1: Breach of No Shop required to be willful, material and/or intentional
    		ANSWER 1: No
    		ANSWER 2: Yes
    TEXT_TYPE: Breach of Meeting Covenant
    	QUESTION 1: Breach of Meeting Covenant required to be willful, material and/or intentional
    		ANSWER 1: No
    		ANSWER 2: Yes
CATEGORY: Remedies
    TEXT_TYPE: Specific Performance
    	QUESTION 1: Specific Performance-Answer
    		ANSWER 1: \"entitled to seek\" specific performance
    		ANSWER 2: \"entitled to\" specific performance
"""


class MAUD(AbstractDataset):
    def __init__(self):
        super().__init__("MAUD", "https://huggingface.co/datasets/theatticusproject/maud")

    def get_data(self):
        def read_info_str_to_dict():
            """This function reads the string in INFO_STR and parses it into a dict"""
            info_dict = {}
            for line in INFO_STR.splitlines():
                line = line.strip()
                if line.startswith("CATEGORY:"):
                    category = line.split(":")[1].strip()
                    info_dict[category] = {}
                elif line.startswith("TEXT_TYPE:"):
                    text_type = line.split(":")[1].strip()
                    info_dict[category][text_type] = {}
                elif line.startswith("QUESTION"):
                    question = line.split(":")[1].strip()
                    info_dict[category][text_type][question] = []
                elif line.startswith("ANSWER"):
                    answer = line.split(":")[1].strip()
                    info_dict[category][text_type][question].append(answer)
            return info_dict

        info_dict = read_info_str_to_dict()

        df = load_dataset("theatticusproject/maud", "maud", split="train")

        instruction_bank = [
            "Consider the following deal point text from a US merger agreement.",
            "Look at the following deal point text from a US merger agreement.",
        ]
        jurisdiction = JURISDICTION.US
        prompt_language = "en"
        answer_language = "en"

        for example in df:
            category = example["category"]
            text_type = example["text_type"]
            question = example["question"]
            # we don't need this because the multilabel one is not really multilabel
            subquestion = example["subquestion"]
            answer = example["answer"]
            answers_lookup = info_dict[category][text_type][question]

            task_type = TASK_TYPE.TEXT_CLASSIFICATION
            text = f"{self.random.choice(instruction_bank)}.\n\n{example['text']}\nWhat is the ABA category?\n{category}"
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)

            text = f"{self.random.choice(instruction_bank)}.\n\n{example['text']}\nWhat is the ABA text type?\n{text_type}"
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)

            text = f"{self.random.choice(instruction_bank)}.\n\n{example['text']}\nWhat is the ABA question?\n{question}"
            yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)

            # do not distinguish between multiple choice and multilabel, since the multilabel ones do not seem to be really multilabel
            if answer in answers_lookup:
                task_type = TASK_TYPE.MULTIPLE_CHOICE
                text = f"{self.random.choice(instruction_bank)}\n\n" \
                       f"{example['text']}\n\n" \
                       f"Answer this question: {question}\n\n" \
                       f"Possible answers: {','.join([f'{idx}: {answer}' for idx, answer in enumerate(answers_lookup)])}\n" \
                       f"Correct answer: {answers_lookup.index(answer)}: {answer}"
                yield self.build_data_point(prompt_language, answer_language, text, task_type, jurisdiction)
