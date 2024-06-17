from typing import List, Dict

from baselines.core.constants import CONTENT
from baselines.mappers.core_utils import split_paragraphs, split_words


def line_counter(text: str, paragraph_end: str = '\n', remove_empty: bool = True) -> int:
    '''
    A wrapper over split_paragraphs, returning the number of line/paragraphs in a page.
    @param text: The text to split.
    @param paragraph_end: The sequence of character that marks a line/paragraph end. default: '\n' is used for lines, but can be '\n\n' for paragraphs.
    @param remove_empty: Whether to remove empty lines/paragraphs.
    Returns: The number of lines/paragraphs in the text.
    >>> line_counter('Hello\n\nWorld and all\nand beyond')
    3
    '''
    return len(split_paragraphs(text, paragraph_end, remove_empty))


def line_counter_enricher(page: Dict, paragraph_end: str = '\n', remove_empty: bool = True,
                          key: str = 'num_lines', overwrite: bool = False) -> List[Dict]:
    '''
    Enriches a page with the number of line/paragraphs in the text.
    @param page: The page to enrich.
    @param paragraph_end: The sequence of character that marks a line/paragraph end. default: '\n' is used for lines, but can be '\n\n' for paragraphs.
    @param remove_empty: Whether to remove empty lines/paragraphs.
    @param key: The key to use for the enrichment.
    @param overwrite: Whether to overwrite an existing key.
    Returns: The enriched page, with an additional key 'num_sentences' containing the number of sentences in the text.
    '''
    assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
    page[key] = line_counter(page[CONTENT], paragraph_end, remove_empty)
    return [page]
    
def word_counter_enricher(page: Dict, key: str = 'word_count', overwrite: bool = False, ignore_punctuation=True, **kwargs) -> List[Dict]:
    '''
    Enriches a page with the number of words in the text.
    @param page: The page to enrich.
    @param key: The key to use for the enrichment.
    @param overwrite: Whether to overwrite an existing key.
    @kwargs: Extra arguments passed into split_words such as model and ignore_whitespace. The default for ignore_punctuation is set to be True for more
    accurate word counts.  
    
    Returns: The enriched page, with an additional key 'num_sentences' containing the number of sentences in the text.
    '''
    assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
    page[key] = len(split_words(page[CONTENT], ignore_punctuation=ignore_punctuation,**kwargs))
    return [page]
