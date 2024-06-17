# Most of the code is copied/adapted from Dolma: https://github.com/allenai/dolma/blob/main/python/dolma/core/utils.py
import re
from typing import List
from datetime import datetime
import hashlib

import fasttext
from uniseg.wordbreak import words

try:
    import blingfire

    BLINGFIRE_AVAILABLE = True
except Exception:
    BLINGFIRE_AVAILABLE = False

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from unidecode import unidecode

def do_once(func):
    """
    A decorator that runs a function only once.
    """
    done = set()

    def wrapper(*args, **kwargs):
        fn_str_parts = [func.__name__]
        if len(args) > 0:
            args = [str(a) for a in args]
            args_str = "_".join(args)
            fn_str_parts.append(args_str)
        if len(kwargs) > 0:
            sorted_kwargs = sorted(kwargs.keys())
            kwargs_str = "_".join([f'{k}:{kwargs[k]}' for k in sorted_kwargs])
            fn_str_parts.append(kwargs_str)
        name = "|".join(fn_str_parts)    
        
        if name not in done:
            done.add(name)
            return func(*args, **kwargs)

    return wrapper

sent_tokenizer = None

@do_once
def _prep_nltk_tokenizer(tokenizer_lang):
    try:
        nltk.data.find('tokenizers/punkt')
        global sent_tokenizer
        if tokenizer_lang in ['en', 'english']:
            # This is the specific tokenizer used in C4
            sent_tokenizer = nltk.data.load("nltk:tokenizers/punkt/english.pickle")
        else:
            sent_tokenizer = PunktSentenceTokenizer()
    except LookupError:
        nltk.download('punkt')


def split_paragraphs(text: str, paragraph_end='\n', remove_empty: bool = True) -> List[str]:
    r"""
    Split a string into paragraphs. A paragraph is defined as a sequence of zero or more characters, followed
    by a newline character(s), or a sequence of one or more characters, followed by the end of the string.

    Args:
        text (str): The text to split.
        paragraph_end (str): The sequence of character that marks a paragraph end. default: '\n', but can be '\n\n' for example.
        remove_empty (bool): Whether to remove empty paragraphs.

    Returns:
        List[str]: A list of strings where each string represents a paragraph from the original text.
    """
    paragraphs = re.split(paragraph_end, text)
    if remove_empty is True:
        paragraphs = [p for par in paragraphs if (p := par.strip())]
    return paragraphs


def split_sentences(text: str, remove_empty: bool = True, tokenizer='blingfire', tokenizer_lang=None) -> List[str]:
    """
    Split a string into sentences.
    Note - this is not perfect (as you can see by the e.g. example below)

    Args:
        text (str): The text to split.
        remove_empty (bool): Whether to remove empty sentences.
        tokenizer (str): The tokenizer to use. one of 'blingfire' or 'nltk'. default: 'blingfire'

    Returns:
        List[str]: A list of strings where each string represents a sentence from the original text.

    Raises:
        ValueError: If an unknown sentence tokenizer is specified.
    """
    if len(text) == 0:
        return []
    if tokenizer == 'blingfire':
        assert BLINGFIRE_AVAILABLE, "Blingfire is not available. Please install it with `pip install blingfire`"
        _, offsets = blingfire.text_to_sentences_and_offsets(text)
    elif tokenizer == 'nltk':
        _prep_nltk_tokenizer(tokenizer_lang)
        offsets = [(start, end) for start, end in sent_tokenizer.span_tokenize(text)]
    else:
        raise ValueError(f"Unknown sentence tokenizer: {tokenizer}")

    if remove_empty is True:
        return [t for start, end in offsets if (t := text[start:end].strip())]
    else:
        raise NotImplementedError("remove_empty=False is not implemented yet")


def split_words(text: str, model='fasttext', ignore_punctuation: bool = False, ignore_whitespace: bool = True) -> \
        List[str]:
    """
    Counts the number of words in a text string.

    Args:
        text (str): The text to count words in.
        model (str): The tokenizer model to use. one of 'uniseg' and 'fasttext'.
        ignore_punctuation (bool): Whether to ignore punctuation.
        ignore_whitespace (bool): Whether to ignore whitespace.

    Returns:
        List[str]: A list of strings where each string represents a word from the original text.

    Raises:
        ValueError: If an unknown word tokenizer model is specified.
    """
    if model == 'uniseg':
        tokens = words(text)
    elif model == 'fasttext':
        tokens = fasttext.FastText.tokenize(text)
    elif model == 'split':
        tokens = text.split()
    else:
        raise ValueError(f"Unknown word tokenizer: {model}")

    if ignore_punctuation and ignore_whitespace:
        return list(w for w in tokens if w[0].isalnum())
    elif ignore_punctuation:
        return list(w for w in tokens if w[0].isalnum() or w[0].isspace())
    elif ignore_whitespace:
        return list(w for w in tokens if w.strip())
    else:
        return list(tokens)


def join_sentences(lines: List[str], sep=' ') -> str:
    """
    Join a list of sentences into a single string (paragraph).

    Args:
        lines (List[str]): The list of sentences to join.
        sep (str): The separator to use.

    Returns:
        str: A single string made by joining the input list of strings.
    """
    return sep.join(lines)


def join_paragraphs(paragraphs: List[str], sep='\n') -> str:
    """
    Join a list of paragraphs into a single string.

    Args:
        paragraphs (List[str]): The list of paragraphs to join.
        sep (str): The separator to use.

    Returns:
        str: A single string made by joining the input list of strings.
    """
    return sep.join(paragraphs)


def normalize_url(url: str) -> str:
    """
    Normalizes urls as a way to assist with dedup. The specific rule is taken
    from the TFDS C4 repo: 

    https://github.com/tensorflow/datasets/blob/fbacae9034d61870ae8d639c7d3f4a667c434879/
    tensorflow_datasets/text/c4_utils.py#L501

    Args:
        url (str): The url to normalize

    Returns:
        str: A normalized url 

    """
    url = re.sub(r"https?:\/\/(www\.)?", "", url)
    url = re.sub(r"\?(utm_|ref|feed).*", "", url)
    url = url.rstrip("/")
    
    return url


def normalize_whitespace_and_lowercase(text: str) -> str:
    """
    Normalizes paragraphs by stripping whitespace and converting to lowercase. 

    Args:
        text (str): The text to normalize

    Returns:
        str: A string that is case and {leading, trailing} whitespace-normalized

    """
    return text.strip().lower()


def normalize_timestamps(timestamp: str, date_format = '%Y-%m-%dT%H:%M:%SZ', default_val=-1.0) -> float:
    """
    Converts timestamp strings into a float which can be used for comparisons.

    Args:
        timestamp (str): The timestamp to convert
        date_format (str): The string specifying the format for the timestamp
        default_val (float): The value assigned to any timestamp where the timestamp and date_format
        are not compatible with each other. 

    Returns:
        float: A numeric representation of the timestamp

    """
    try:
        return datetime.strptime(timestamp, date_format).timestamp()
    except ValueError:
        return default_val


def hash_text(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

UNICODE_PUNCT = {
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

UNICODE_PUNCT_RE = re.compile(f"[{''.join(UNICODE_PUNCT.keys())}]")
NON_PRINTING_CHARS_RE = re.compile(f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]")
DIGIT_RE = re.compile(r"\d")
PUNCT_OR_NON_PRINTING_CHARS_RE = re.compile(
    (UNICODE_PUNCT_RE.pattern + NON_PRINTING_CHARS_RE.pattern).replace("][", "")
)


def ccnet_dedup_normalizer(line: str) -> str:
    """
    Normalizes the string by removing punctuation and non-printable characters and accent characters and replacing all digits with 0.

    Args:
        line (str): string to convert

    Returns:
        str: A normalized string

    """
    line = line.strip()
    if not line:
        return line
    # case
    line = line.lower()
    # numbers
    line = DIGIT_RE.sub("0", line)
    line = PUNCT_OR_NON_PRINTING_CHARS_RE.sub("", line)
    line = unidecode(line)
    return line


DEDUP_NORMALIZERS = {
    'normalize_url': normalize_url,
    'normalize_timestamps': normalize_timestamps,
    'normalize_whitespace_and_lowercase': normalize_whitespace_and_lowercase,
    'hash_text': hash_text,
    'ccnet_dedup_normalizer': ccnet_dedup_normalizer
}
