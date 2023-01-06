from instruction_datasets.brcad_5 import BrCAD5
from instruction_datasets.contract_nli import ContractNLI
from instruction_datasets.eoir_privacy import EOIRPrivacy
from instruction_datasets.german_ler import GermanLER
from instruction_datasets.legal_case_document_summarization import LegalCaseDocumentSummarization
from instruction_datasets.lex_glue import LexGLUE
from instruction_datasets.lextreme import LEXTREME
from instruction_datasets.mining_legal_arguments import MiningLegalArguments
from instruction_datasets.multi_lex_sum import MultiLexSum
from instruction_datasets.olc_memos import OLCMemos
from instruction_datasets.plain_english_contracts_summarization import PlainEnglishContractsSummarization
from instruction_datasets.reddit_legal_qa import RedditLegalQA
from instruction_datasets.short_answer_feedback import ShortAnswerFeedback
from instruction_datasets.swiss_judgment_prediction import SwissJudgmentPrediction
from instruction_datasets.us_class_actions import USClassActions

datasets_to_build = [USClassActions, LEXTREME, LexGLUE, SwissJudgmentPrediction, MultiLexSum,
                     LegalCaseDocumentSummarization, PlainEnglishContractsSummarization, GermanLER,
                     MiningLegalArguments, ContractNLI, ShortAnswerFeedback, EOIRPrivacy, OLCMemos, RedditLegalQA]


# TODO debug this later
# datasets_to_build = [BrCAD5]

def build_instruction_datasets():
    for dataset in datasets_to_build:
        dataset().build_instruction_dataset()


if __name__ == '__main__':
    build_instruction_datasets()
