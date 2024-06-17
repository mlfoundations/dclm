import re
from typing import Dict, List

from core.constants import CONTENT, CHUNK
from core.factory_utils import factory_function


def min_character_filter(page: Dict, min_characters: int = 20) -> List[Dict]:
    """
    Filters the input JSON object based on the number of characters in the CONTENT field.

    This function returns a list containing the input JSON object if the number of characters
    in the CONTENT field is greater than or equal to `min_characters`. Otherwise, it returns
    an empty list.

    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    min_characters -- The minimum number of characters required for the input JSON object
                      to pass the filter.

    Returns:
    A list containing the input JSON object if it passes the filter, or an empty list
    if it doesn't.
    """
    if len(page[CONTENT]) >= min_characters:
        return [page]

    return []


def mock_text_type_enricher(page: Dict, key: str = "text_type", overwrite: bool = False) -> List[Dict]:
    """
    Enriches the input JSON object with a new field that indicates the type of text.

    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    key -- The name of the new field to be added to the input JSON object.
    overwrite -- Whether to overwrite the new field if it already exists in the input JSON object.

    Returns:
    A list containing the input JSON object with the new field added.

    """
    assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
    if "<javascript>" in page[CONTENT]:
        page[key] = "code"
    else:
        page[key] = "text"
    return [page]


@factory_function
def pattern_splitter(pattern: str):
    """
    Splits the input JSON object into multiple JSON objects based on a given pattern. The pattern will be included in
    both chunks around it (i.e. they will overlap, one ending in the pattern and one starting with it).
    """
    pattern = re.compile(pattern)

    def split_by_pattern(page: Dict) -> List[Dict]:
        new_pages = []
        start = 0
        if pattern.search(page[CONTENT]) is None:
            return [page]
        for i, match in enumerate(pattern.finditer(page[CONTENT])):
            new_page = page.copy()
            new_page[CHUNK] = i
            new_page[CONTENT] = page[CONTENT][start:match.end()]
            start = match.start()
            new_pages.append(new_page)
        new_page = page.copy()
        new_page[CHUNK] = len(new_pages)
        new_page[CONTENT] = page[CONTENT][start:]
        new_pages.append(new_page)
        return new_pages
    return split_by_pattern
