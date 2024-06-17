from typing import List, Dict, Union
import re
from urllib.parse import urlparse
from retrie.retrie import Blacklist
import pickle
import random

from core.factory_utils import factory_function
from baselines.core.constants import *


def random_sampling_filter(page, keep_probability=0.1):
    """
    Filter the JSON objects randomly based on a random coinflip, in order to subsample according to a specified probability. 

    Arguments:
    page -- A dictionary representation of the page.
    keep_probability -- What proportion of pages to keep
    Returns:
    A list containing the page if the language is in the keep_languages list and exceeds the threshold, otherwise an empty list.
    """
    assert 0 <= keep_probability <= 1
    return [page] if random.random() < keep_probability else []
    

def language_filter(page: Dict, keep_languages: List[str], key='language_id_whole_page_langdetect', threshold=0.0) -> \
        List[Dict]:
    """
    Filter the JSON objects by keeping only the ones that have the specified language, with the option to provide a threshold on
    the predicted probability

    Arguments:
    page -- A dictionary representation of the page.
    keep_languages -- A list of languages to keep.
    key -- The metadata key for LID, defaults to the key for using langdetect on the whole page
    threshold -- A probability threshold for detecting a language

    Returns:
    A list containing the page if the language is in the keep_languages list and exceeds the threshold, otherwise an empty list.
    """
    if not isinstance(keep_languages, list):
        raise TypeError("The keep_languages argument must be a list.")

    assert key in page, f'The input JSON object does not have a {key} field'
    for lang in keep_languages:
        if lang in page[key] and page[key][lang] > threshold:
            return [page]
    else:
        return []


def quality_filter(page: Dict, key: str = 'fasttext_hq_prob', threshold: float=0.0, lower_better: bool=False, key_must_exist: bool=True) -> List[Dict]:
    """
    Filters the JSON objects based on a quality score (e.g. from a model-based prediction). 

    Arguments: 
    page -- A dictionary representation of the page. 
    key -- A string specifying which quality score, the default is the default key produced by the fasttext hq_prob model
    threshold -- A float indicating the minimum quality required to keep the page. 
    lower_better - A bool for whether lower quality score is better (e.g., for perplexity, lower_better should be True).
    key_must_exist - A bool for whether the key must exist for all pages. If False, will filter out pages that are missing the key
    Returns:
    A list containing the page if the quality exceeds the threshold (or does not when lower_better = True), otherwise an empty list.
    """

    if key_must_exist:
        assert key in page, f'The input JSON object does not have a {key} field'
        quality_score = page[key]
    else:
        missing_score = float('inf') if lower_better else -float('inf')
        quality_score = page.get(key, missing_score)

    if lower_better:
        return [page] if quality_score <= threshold else []
    else:
        return [page] if quality_score >= threshold else [] 


@factory_function
def url_substring_filter(banlist: Union[str, List] = None, banlist_from_fname: str = None, ignore_chars: List[str] = None, 
                         num_banned_substrs: int = 1, exact_domain_match: bool=False, match_substrings=True, case_sensitive=False) -> List[Dict]:
    """
    Filters the input JSON object by URL

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'url' field
            that contains the urls for the pages to be analyzed
    banlist -- A list of banned substrs to look for within a url.
    banlist_from_fname -- Gives the option to load in a large banlist from a .txt file where each substring
                          is on a spearate line. It can also take in a .pkl file containing a pre-compiled regex
                          This takes precedence over passing in via banlist 
    ignore_chars -- A list of characters to ignore (e.g., ['.', "-"]) as they are typically used to bypass
            detectors for fradulent/inappropriate webpages
    num_banned_substrs -- Number of num_banned_substrs within the banlist that must be present
            to be filtered out. Refinedweb uses this for "softer" banlist items (e.g., "webcam", "escort")
    exact_domain_match -- Whether to extract the domain from the page url and check for an exact match (e.g., when
    set to False, "le.com" being in banlist would lead to "google.com" being banned)
    match_substrings -- When True, the banlist items only need to be a substring. When False, items must exist 
            in between word boundaries. Note this is only used when exact_domain_match is False. 
    case_sensitive -- Whether to check for case sensitivity (RefinedWeb sets this to be True)

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """

    # TODO: Right now initialization/compilation for exact_domain_match=False + large banlists (3 mins)
    # Should verify whether we can use exact_domain_match=True 

    if banlist_from_fname is not None and any(banlist_from_fname.endswith(e) for e in ['.pkl', '.pickle']):
        assert not exact_domain_match, "pickled banlist cannot be used with exact_domain_match"
        with open(banlist_from_fname, "rb") as file:
            pattern = pickle.load(file)
    else:
        if banlist_from_fname is not None:        
            with open(banlist_from_fname, "r") as file:
                    banlist = file.read().splitlines()
        elif isinstance(banlist, str):
            banlist = [banlist]

        banlist = [b.lower() for b in banlist] if not case_sensitive else [b for b in banlist]
        if exact_domain_match: 
            banlist = set(banlist)
        else:
            re_flags = re.IGNORECASE if not case_sensitive else None
            pattern = re.compile(Blacklist(banlist, match_substrings=match_substrings, re_flags=re_flags).compiled)

    ignore_chars = [] if ignore_chars is None else ignore_chars

    def filter_fn(page: Dict):
        url = urlparse(page[URL]).netloc if exact_domain_match else page[URL]
        url = url.lower() if not case_sensitive else url

        for char in ignore_chars:
            url = url.replace(char, "")
            
        if exact_domain_match and url in banlist:
           return []
        elif not exact_domain_match:
           banned_subtrs = len(set(pattern.findall(url)))
           if banned_subtrs >= num_banned_substrs:
                return []
        
        return [page]

    return filter_fn