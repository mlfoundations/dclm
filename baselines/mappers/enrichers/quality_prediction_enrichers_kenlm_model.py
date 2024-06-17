''' This is the script from: https://huggingface.co/edugp/kenlm/blob/main/model.py

The `setup.py` downlods two models to the "wikipedia" directory:
1. .bin file (~4GB): https://huggingface.co/edugp/kenlm/blob/main/wikipedia/en.arpa.bin
2. .model file : https://huggingface.co/edugp/kenlm/blob/main/wikipedia/en.sp.model

To install necessary dependencies:
1. pip install https://github.com/kpu/kenlm/archive/master.zip
2. pip install sentencepiece

The following example demonstrates how to calculate perplexity using the KenLM library.
Perplexity is a measurement of how well a probability model predicts a sample.
A lower perplexity indicates the probability distribution is a good model for the data.
In this script, we utilize a pre-trained KenLM model trained on English Wikipedia corpus.

# Load the KenLM model trained on English wikipedia
model = KenlmModel.from_pretrained("wikipedia", "en")

# Calculate and print the perplexity for a formal sentence with correct grammar
print(model.get_perplexity("I am very perplexed"))
# Output: 341.3 (low perplexity, since sentence style is formal and with no grammar mistakes)

# Calculate and print the perplexity for a colloquial sentence with grammar mistakes
print(model.get_perplexity("im hella trippin"))
# Output: 46793.5 (high perplexity, since the sentence is colloquial and contains grammar mistakes)

'''

import os
import re
import unicodedata
from typing import Dict, List, Callable

import kenlm
import sentencepiece

from core.constants import CONTENT
from core.factory_utils import factory_function

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_SUBDIRECTORY = "baselines/mappers/enrichers/quality_prediction_enrichment_models"


class SentencePiece:
    def __init__(
            self,
            model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: dict) -> dict:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"
    )
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    def __init__(
            self,
            model_dataset: str,
            language: str,
            lower_case: bool = False,
            remove_accents: bool = False,
            normalize_numbers: bool = True,
            punctuation: int = 1,
    ):
        if os.path.exists(MODEL_SUBDIRECTORY):
            model_dir = MODEL_SUBDIRECTORY
        else: 
            model_dir = os.path.join(PROJECT_ROOT, MODEL_SUBDIRECTORY)

        self.model = kenlm.Model(os.path.join(model_dir, f"{language}.arpa.bin"))
        self.tokenizer = SentencePiece(os.path.join(model_dir, f"{language}.sp.model"))
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation

    @classmethod
    def from_pretrained(
            cls,
            model_dataset: str,
            language: str,
    ):
        return cls(
            model_dataset,
            language,
            False,
            False,
            True,
            1,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
            self,
            line: str,
            accent: bool = True,
            case: bool = True,
            numbers: bool = True,
            punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    def strip_accents(self, line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)


@factory_function
def ken_lm_perplexity_enricher(key: str = "kenlm_perplexity", overwrite: bool = False) -> Callable[[Dict], List[Dict]]:
    '''
    Enriches a page with the perplexity of its content.
    :param key: the key to use for storing the perplexity
    :param overwrite: whether to overwrite an existing key
    :return: a function that enriches a page
    '''

    ken_lm_model = KenlmModel.from_pretrained("wikipedia", "en")

    def enrich(page: Dict) -> List[Dict]:
        assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
        page[key] = ken_lm_model.get_perplexity(page[CONTENT])
        return [page]

    return enrich
