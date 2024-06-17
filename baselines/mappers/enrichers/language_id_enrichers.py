'''
This script uses a pre-trained FastText model to predict the language of a given text. The model used is 'lid.176.bin', which
can be downloaded from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

This script requires the fasttext library. You can install it with 'pip install fasttext'.
'''

import string
from typing import List, Dict, Callable, Union

from langdetect import detect_langs as detect_langdetect

from core.constants import CONTENT
from baselines.mappers.core_utils import split_sentences
from core.factory_utils import factory_function

FASTTEXT = 'fasttext'
LANGDETECT = 'langdetect'

# Mapping of models to their respective detection functions
import os
import fasttext

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_SUBDIRECTORY = "baselines/mappers/enrichers/language_id_enrichment_models"
FASTTEXT_MODEL_FILENAME = "lid.176.bin"
MODEL_DIRECTORY = os.path.join(PROJECT_ROOT, MODEL_SUBDIRECTORY)


def load_fasttext_model():
    if os.path.exists(MODEL_SUBDIRECTORY):
        model_path = os.path.join(MODEL_SUBDIRECTORY, FASTTEXT_MODEL_FILENAME)
    else:
        model_path = os.path.join(MODEL_DIRECTORY, FASTTEXT_MODEL_FILENAME)

    return fasttext.load_model(model_path)

def get_fasttext_lang_prob(model: fasttext.FastText._FastText, text: str) -> (str, float):
    '''
    Function to detect the language of a given text using FastText model.

    Parameters:
        model (fasttext.FastText._FastText): The FastText model to use for language detection.
        text (str): The text whose language is to be detected.

    Returns:
        str: The detected language.
        prob: The probability of the detected language.
    '''
    # Get the language prediction from the model
    predictions = model.predict(text)

    # Extract the language label, and remove the "__label__" prefix
    lang = predictions[0][0].replace("__label__", "")
    prob = predictions[1][0]

    # Return the detected language
    return {lang: prob}


def get_langdetect_lang_prob(text: str) -> (str, float):
    detected_lang = detect_langdetect(text)
    return {x.lang: x.prob for x in detected_lang}


def is_space_or_punct(s: str) -> bool:
    '''
    Check if a string is empty, or contains only spaces or punctuation.

    Parameters:
        s (str): The string to check.

    Returns:
        bool: True if the string is empty, or contains only spaces or punctuation, otherwise False.
    '''
    punct = set(string.punctuation)
    for char in s:
        if char not in punct and char != ' ':
            return False
    return True


def detect_lang_paragraph_helper(text: str, detect_func: Callable, tokenizer: str = 'blingfire', *args) -> Dict[
    str, Dict[str, Union[float, int]]]:
    '''
    Detects the language(s) of a given text paragraph.

    This function splits the text into individual sentences. It then employs the given detection function to identify the
    language of each sentence. Subsequently, the probabilities are averaged for each language across all sentences,
     and the output includes the count of sentences for each detected language.

    Parameters:
        text (str): The paragraph of text to detect the language for.
        detect_func (Callable): The function used to detect the language of a sentence.
        tokenizer (str): The method or tool used for tokenizing the text into sentences. Currently supports 'blingfire' and 'nltk', defaults to 'blingfire'.
        *args: Additional arguments to pass to the detect function.

    Returns: Dict[str, List[float]]: A dictionary with languages as keys and probability averages and number of
    occurrences as values. Returns an empty dictionary if the text is empty or contains only spaces or punctuation.
    '''
    all_languages = {}
    for sent in split_sentences(text, tokenizer=tokenizer):
        if not is_space_or_punct(sent):
            lang_dict = detect_func(*args, sent)

            # Update the probabilities for each language
            for lang, prob in lang_dict.items():
                if lang in all_languages:
                    all_languages[lang].append(prob)
                else:
                    all_languages[lang] = [prob]

    # Reduce the probabilities to average probabilities and sentence counts
    reduced_languages = reduce_language_probabilities(all_languages)
    return reduced_languages


def reduce_language_probabilities(lang_dict: Dict[str, List[float]]) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Reduces the language probabilities dictionary to average probabilities and sentence counts.

    Parameters:
        lang_dict (Dict[str, List[float]]): Dictionary containing languages and their probabilities.

    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Reduced dictionary with average probabilities and sentence counts.
    """
    reduced_dict = {}
    for lang, probs in lang_dict.items():
        total_probs = sum(probs)
        avg_prob = total_probs / len(probs)
        reduced_dict[lang] = {
            'average_probability': avg_prob,
            'number_of_sentences': len(probs)
        }
    return reduced_dict


def detect_lang_whole_page_langdetect(text: str, seed=None) -> Dict[str, float]:
    '''
    Detect the language of an entire page using the langdetect model, without splitting into sentences first

    Parameters:
        text (str): The text whose languages are to be detected.
        seed (Otpional[int]): The seed to use to make this deterministic.

    Returns:
        Dict where keys are strings specifying individual languages and values are predicted probabilities/
        Returns an empty list if the text is empty or contains only spaces or punctuation.
    '''
    if not is_space_or_punct(text):
        if seed is not None:
            from langdetect import DetectorFactory
            DetectorFactory.seed = seed  # https://snyk.io/advisor/python/langdetect/functions/langdetect.DetectorFactory.seed

        lang_dict = get_langdetect_lang_prob(text)
    else:
        lang_dict = {}
    return lang_dict


def detect_lang_whole_page_fasttext(model: fasttext.FastText._FastText, text: str, seed=None) -> List[str]:
    '''
    Detect the language of an entire page using the fasttext model, without splitting into sentences first

    Parameters:
        model (fasttext.FastText._FastText): The FastText model to use for language detection.
        text (str): The text whose languages are to be detected.

    Returns:
        Returns: Dict[str, List[float]]: A dictionary with languages as keys and predicted probabilities as values.
        Returns an empty dictionary if the text is empty or contains only spaces or punctuation.
    '''
    if seed is not None:
        raise NotImplementedError(f'Setting seed for fast detect is not implemented')
    if not is_space_or_punct(text):
        lang_dict = get_fasttext_lang_prob(model, text.replace("\n", ""))
    else:
        lang_dict = {}
    return lang_dict


@factory_function
def detect_lang_whole_page_enricher(model: str, key_prefix: str = "language_id_whole_page",
                                    overwrite: bool = False, seed=None) -> Callable[[Dict], List[Dict]]:
    '''
    Enrichers a page with language detection information based upon all the text in the whole page. Using the fasttext model,
    it is necessary to remove extra newline characters before performing inference.

    Parameters:
        model (str): The model to use for detection. Currently supports 'fasttext' and 'langdetect'.
        key (str): The key to use for storing the language detection information in the page.
        overwrite (bool): Whether to overwrite the existing language detection information in the page.
        seed (Otpional[int]): The seed to use to make this deterministic.

    Returns:
        A function that enriches a page with language detection information.
    '''

    key = f"{key_prefix}_{model}"
    if model == FASTTEXT:
        # only load it if needed
        fasttext_model = load_fasttext_model()
    elif model == LANGDETECT and seed is not None:
        if seed is not None:
            from langdetect import DetectorFactory
            DetectorFactory.seed = seed  # https://snyk.io/advisor/python/langdetect/functions/langdetect.DetectorFactory.seed

        seed = None  # only set the seed once
    else:
        fasttext_model = None

    def enrich(page: Dict) -> List[Dict]:
        assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
        if model == FASTTEXT:
            page[key] = detect_lang_whole_page_fasttext(fasttext_model, page[CONTENT], seed=seed)
        elif model == LANGDETECT:
            page[key] = detect_lang_whole_page_langdetect(page[CONTENT], seed=seed)
        else:
            raise ValueError(f"model {model} is not supported")
        return [page]

    return enrich


@factory_function
def detect_lang_paragraph_enricher(model: str, tokenizer: str, key_prefix: str = "language_id_paragraph",
                                   overwrite: bool = False) -> Callable[[Dict], List[Dict]]:
    '''
    Enrichers the page with language detection information.
    This function uses the detect_lang_paragraph_helper that splits the given paragraph to sentences,
    detects the language of each sentence, and returns the average probabilities and sentence counts for each language.

    Parameters:
        model (str): The model to use for detection. Currently supports 'fasttext' and 'langdetect'.
        tokenizer (str): The tokenizer to use for splitting the text into sentences. Currently supports 'blingfire' and 'nltk'.
        key (str): The key to use for storing the language detection information in the page.
        overwrite (bool): Whether to overwrite the existing language detection information in the page.

    Returns:
        A function that enriches a page with language detection information.
    '''
    key = f"{key_prefix}_{model}"
    fasttext_model = load_fasttext_model()
    
    def enrich(page: Dict) -> List[Dict]:
        assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
        if model == FASTTEXT:
            page[key] = detect_lang_paragraph_helper(page[CONTENT], get_fasttext_lang_prob, tokenizer,
                                                     fasttext_model)
        elif model == LANGDETECT:
            page[key] = detect_lang_paragraph_helper(page[CONTENT], get_langdetect_lang_prob, tokenizer)
        else:
            raise ValueError(f"model {model} is not supported")
        return [page]

    return enrich