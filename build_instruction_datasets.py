"""Main script to build all instruction datasets.

Usage:
    python build_instruction_datasets.py --datasets BrazilianBarExam BrCAD5
"""
import argparse
import functools
import multiprocessing
from typing import Optional, Type, Sequence

from abstract_dataset import AbstractDataset
from instruction_datasets.brazilan_bar_exam import BrazilianBarExam
from instruction_datasets.brcad_5 import BrCAD5
from instruction_datasets.bva_decisions import BVADecisions
from instruction_datasets.ca_bar_exam_essays import CABarExamEssays
from instruction_datasets.cail_2019 import CAIL2019
from instruction_datasets.cail_2022 import CAIL2022
from instruction_datasets.case_briefs import CaseBriefs
from instruction_datasets.change_my_view import ChangeMyView
from instruction_datasets.civipro_questions import CiviproQuestions
from instruction_datasets.coliee import COLIEE
from instruction_datasets.contract_nli import ContractNLI
from instruction_datasets.edgar_ner import EdgarNER
from instruction_datasets.eoir_privacy import EOIRPrivacy
from instruction_datasets.eur_lex_sum import EurLexSum
from instruction_datasets.german_ler import GermanLER
from instruction_datasets.german_rental_agreements import \
    GermanRentalAgreements
from instruction_datasets.greek_ner import Ell18Dataset
from instruction_datasets.greek_ner import Ell4Dataset
from instruction_datasets.gsm8k import GSM8K
from instruction_datasets.ildc import ILDC
from instruction_datasets.indian_ner import IndianNER
from instruction_datasets.indian_text_segmentation import \
    IndianTextSegmentation
from instruction_datasets.international_citizenship_law_questions import \
    InternationalCitizenshipLawQuestions
from instruction_datasets.jec_qa import JECQA
from instruction_datasets.korean_legal_qa import KoreanLegalQA
from instruction_datasets.lbox_open import LboxOpen
from instruction_datasets.legal_case_document_summarization import \
    LegalCaseDocumentSummarization
from instruction_datasets.legal_qa import LegalQA
from instruction_datasets.lex_glue import LexGLUE
from instruction_datasets.lextreme import LEXTREME
from instruction_datasets.lila import Lila
from instruction_datasets.littleton import Littleton
from instruction_datasets.logi_qa import LogiQA
from instruction_datasets.maud import MAUD
from instruction_datasets.mbe import MBE
from instruction_datasets.mc_exams_law import MCExamsLaw
from instruction_datasets.mining_legal_arguments import MiningLegalArguments
from instruction_datasets.multi_lex_sum import MultiLexSum
from instruction_datasets.natural_instructions_legal import \
    NaturalInstructionsLegal
from instruction_datasets.natural_instructions_other import \
    NaturalInstructionsOther
from instruction_datasets.olc_memos import OLCMemos
from instruction_datasets.plain_english_contracts_summarization import \
    PlainEnglishContractsSummarization
from instruction_datasets.privacy_qa import PrivacyQA
from instruction_datasets.privacy_summarization import PrivacySummarization
from instruction_datasets.professional_law import ProfessionalLaw
from instruction_datasets.reclor import ReClor
from instruction_datasets.reddit_legal_qa import RedditLegalQA
from instruction_datasets.sara import Sara
from instruction_datasets.sara_prolog import SaraProlog
from instruction_datasets.short_answer_feedback import ShortAnswerFeedback
from instruction_datasets.spanish_labor_law import SpanishLaborLaw
from instruction_datasets.stack_exchange_questions_legal import \
    StackExchangeQuestionsLegal
from instruction_datasets.swiss_judgment_prediction import \
    SwissJudgmentPrediction
from instruction_datasets.tscc_alqac import TsccAlqac
from instruction_datasets.us_class_actions import USClassActions
from instruction_datasets.valid_wills import ValidWills
from instruction_datasets.xp3mt import XP3MT


legal_datasets = [
    BrazilianBarExam,
    BrCAD5,
    BVADecisions,
    CABarExamEssays,
    CAIL2019,
    CAIL2022,
    CaseBriefs,
    ChangeMyView,
    CiviproQuestions,
    COLIEE,
    ContractNLI,
    EdgarNER,
    Ell4Dataset,
    Ell18Dataset,
    EOIRPrivacy,
    EurLexSum,
    GermanLER,
    GermanRentalAgreements,
    GSM8K,
    ILDC,
    IndianNER,
    IndianTextSegmentation,
    InternationalCitizenshipLawQuestions,
    JECQA,
    KoreanLegalQA,
    LboxOpen,
    LegalCaseDocumentSummarization,
    LegalQA,
    LexGLUE,
    LEXTREME,
    Lila,
    Littleton,
    LogiQA,
    MAUD,
    MBE,
    MCExamsLaw,
    MiningLegalArguments,
    MultiLexSum,
    OLCMemos,
    PlainEnglishContractsSummarization,
    PrivacyQA,
    PrivacySummarization,
    ProfessionalLaw,
    ReClor,
    RedditLegalQA,
    Sara,
    SaraProlog,
    ShortAnswerFeedback,
    SpanishLaborLaw,
    StackExchangeQuestionsLegal,
    SwissJudgmentPrediction,
    TsccAlqac,
    USClassActions,
    ValidWills,
]
natural_instructions = [NaturalInstructionsLegal, NaturalInstructionsOther]
xp3mt = [XP3MT]

erroneous_datasets = []
datasets_already_built = [
    BrazilianBarExam, BrCAD5, BVADecisions, CABarExamEssays, CAIL2019, CAIL2022, CaseBriefs, ChangeMyView,
    CiviproQuestions, COLIEE, ContractNLI, EdgarNER, Ell4Dataset, Ell18Dataset, EOIRPrivacy, EurLexSum, GermanLER,
    GermanRentalAgreements, GSM8K, ILDC, IndianNER, IndianTextSegmentation, InternationalCitizenshipLawQuestions, JECQA,
    KoreanLegalQA, LboxOpen, LegalCaseDocumentSummarization, LegalQA, LexGLUE, LEXTREME, Lila, Littleton, LogiQA, MAUD,
    MBE, MCExamsLaw, MiningLegalArguments, MultiLexSum, OLCMemos, PlainEnglishContractsSummarization, PrivacyQA,
    PrivacySummarization, ProfessionalLaw, ReClor, RedditLegalQA, Sara, SaraProlog, ShortAnswerFeedback,
    SpanishLaborLaw, StackExchangeQuestionsLegal, SwissJudgmentPrediction, TsccAlqac, USClassActions, ValidWills,
]
datasets_already_built += natural_instructions
datasets_already_built += xp3mt


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Builds the instruction datasets', )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Builds a small version of the dataset for debugging',
    )
    parser.add_argument(
        '--build_from_scratch',
        action='store_true',
        help='Builds the dataset from scratch, even if it already exists',
    )
    parser.add_argument("--processes",
                        type=int,
                        default=1,
                        help="Number of processes to use")
    parser.add_argument("--datasets",
                        type=str,
                        nargs="+",
                        default=[],
                        help="Datasets to build")
    args = parser.parse_args(args)

    # If no datasets are specified, build all of them
    if not args.datasets:
        args.datasets = [
            dataset.__name__
            for dataset in (legal_datasets + natural_instructions + xp3mt)
        ]
    # Get the actual classes for each named dataset.
    # This is a bit hacky, but it works.
    # TODO(arya): Create a dict of dataset names to classes and use that instead
    args.datasets = [globals()[dataset] for dataset in args.datasets]

    return args


def _build_dataset(dataset: Type[AbstractDataset], debug_size: int) -> None:
    # TODO(arya): We have a Liskov substitution principle violation here.
    #   AbstractDataset's __init__ takes two arguments, but the
    #  __init__s of the subclasses take none. Should rectify, perhaps by
    #  composition instead of inheritance.
    dataset().build_instruction_dataset(debug_size=debug_size)


def build_instruction_datasets(datasets: Sequence[Type[AbstractDataset]],
                               *,
                               processes: int,
                               debug: bool = False,
                               build_from_scratch: bool = False) -> None:
    if debug:
        datasets_to_build = erroneous_datasets
        debug_size = 5
        processes = 1  # Parallelism would only introduce more confusion.
    else:
        datasets_to_build = [
            dataset for dataset in datasets
            if dataset not in erroneous_datasets
        ]
        debug_size = -1

        if not build_from_scratch:
            datasets_to_build = [dataset for dataset in datasets_to_build if dataset not in datasets_already_built]

    build_one = functools.partial(_build_dataset, debug_size=debug_size)

    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(build_one, datasets_to_build)


if __name__ == '__main__':
    args = parse_args()
    build_instruction_datasets(args.datasets,
                               processes=args.processes,
                               debug=args.debug,
                               build_from_scratch=args.build_from_scratch)
