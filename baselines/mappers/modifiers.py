import re
from typing import List, Dict, Union
import warnings

import justext
from lxml.etree import ParserError
from retrie.retrie import Blacklist

from baselines.mappers.core_utils import split_paragraphs, split_words
from core.constants import CONTENT, URL
from core.factory_utils import factory_function
from bs4 import BeautifulSoup
import random
import copy


def starcoder_v2_repo_splitter(page: Dict, max_files=1000, delete_content=True):
    """
    Modifies the input JSON object by splitting the contents of large repos into smaller chunks.

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
                 that contains the text to be analyzed. page corresponds to a single repo
                 where 'content' is a list of dictionaries, each containing 'text' and 'filename'
                 (along with other metadatda)
    
    max_files (int): The maximum number of files to include in each chunk. The default is 1000.
    delete_content (bool): Whether or not to delete the CONTENT field from the resulting pages.              
    """
    if len(page['files']) <= max_files:
        return [page]
    else:
        chunks = []
        for i in range(0, len(page['files']), max_files):
            new_page = copy.deepcopy(page)
            new_page['files'] = page['files'][i:i+max_files]
            if delete_content and CONTENT in new_page:
                del new_page[CONTENT]
            chunks.append(new_page)
        return chunks


def starcoder_v2_format_modifier(page: Dict, add_metadata_prob: float=0.5, add_sentinels: bool=True) -> List[Dict]:
    """
    Modifies the input JSON object by formatting the content to match the StarCoder V2 format. 
    This includes concatenating all files in a repo together and gives the option to add metadata 
    information (i.e., repo name, file name, etc.) to the content.

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
                 that contains the text to be analyzed. page corresponds to a single repo
                 where 'content' is a list of dictionaries, each containing 'text' and 'filename'
                 (along with other metadatda)
    add_metadata_prob (float): Specifies a probability that metadata is added in between file contents. If 0,
                  no metadata is added. If 1, metadata is added between every file. The default is 0.5, 
                  following the original StarCoder V2 paper. 
    add_sentinels (bool): Whether to add special tokens to indicate separations between files as well as the repo name.
                   The default is True following Starcoder V2. 

    Returns:
    List[Dict]: A list containing the input JSON object with the content formatted into a single text string. 
    """

    assert 0 <= add_metadata_prob <= 1

    if random.random() < add_metadata_prob:
        file_texts = [f"<file_sep>{f['filename']}\n{f['text']}" for f in page['files']]
        text = f"<repo_name>{page['repo_name']}" + "".join(file_texts)
    else:
        file_texts = [f"<file_sep>{f['text']}" for f in page['files']]
        text =  "".join(file_texts)

    if not add_sentinels:
        text = text.replace("<file_sep>", "\n\n").replace("<repo_name>", "").strip()
        
    page[CONTENT] = text

    return [page]


@factory_function
def arxiv_appendix_modifier() -> List[Dict]:
    """
    Modify the input JSON object by removing any content after the first occurrence of either \appendix, \bibliography,
    or variations of \begin{references}. 

    Taken from rpj_v1: https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/arxiv_cleaner.py

    Args:
        page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
                     that contains the text to be analyzed.

    Returns:
        List[Dict]: A list containing the input JSON object with content after the specified headers removed.
    """
    
    pattern = r"(\\appendix|\\begin\{references\}|\\begin\{REFERENCES\}|\\begin\{thebibliography\}|\\bibliography\{.*?\}).*$"
    pattern = re.compile(pattern, flags=re.DOTALL)

    def modify(page):
        new_content = pattern.sub('', page[CONTENT])
        
        if new_content == '':
            return []

        page[CONTENT] = new_content
        return [page]

    return modify

@factory_function
def arxiv_comment_modifier(remove_multiline=False) -> List[Dict]:
    """
    Modify the input JSON object by removing LaTeX comments from the content.
    This includes both single-line comments (starting with '%') and multi-line comments
    enclosed within \begin{comment} and \end{comment} tags. It also removes in-line comments
    that are not at the start of a line. 

    Taken from rpj_v1: https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/arxiv_cleaner.py

    Args:
        page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
                     that contains the text to be analyzed.

    Returns:
        List[Dict]: A list containing the input JSON object with all LaTeX comments removed.
    """

    line_pattern =  re.compile(r'(?m)^%.*\n?', flags=re.MULTILINE)
    within_line_pattern = re.compile(r'[^\\]%.+$', flags=re.MULTILINE)
    multiline_pattern = re.compile(r'\\begin{comment}.*?\\end{comment}', flags=re.DOTALL)

    def modify(page):

        # Remove all line comments
        new_content = line_pattern.sub('', page[CONTENT])

        # Remove in-line comments
        new_content = within_line_pattern.sub('', new_content)
   
        # Removes multiline comments (not in official rpj_v1 code)
        if remove_multiline:
            new_content = multiline_pattern.sub('', new_content)

        if new_content == '':
            return []

        page[CONTENT] = new_content
        return [page]

    return modify

@factory_function
def arxiv_macro_modifier() -> List[Dict]:
    """
    Modify the input JSON object by inline-expanding LaTeX macros defined via \newcommand and \def
    that have no arguments. This function targets macros that are defined but not expanded.

    Taken from rpj_v1: https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/arxiv_cleaner.py

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
                 that contains the text to be analyzed.

    Returns:
    List[Dict]: A list containing the input JSON object with specified LaTeX macros inline-expanded.
    """

    # regex for extracting \newcommand macros without arguments
    non_arg_nc_reg = re.compile(
        pattern=r'\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$',
        flags=re.MULTILINE
    )

    # regex for extracting \def macros without arguments
    non_arg_def_reg = re.compile(
        pattern=r'\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$',
        flags=re.MULTILINE
    )

    def modify(page):
        content = page.get(CONTENT, '')

        # Extract all user-defined LaTeX macros from the preamble
        macros = {}
        for reg in [non_arg_nc_reg, non_arg_def_reg]:
            for match in reg.finditer(content):
                macro_name = match \
                    .group(1).encode("unicode-escape").decode("utf-8")
                macro_val = match \
                    .group(2).encode("unicode-escape").decode("utf-8")

                macros[macro_name] = macro_val
                
        # Inline-expand all non-arg macros
        for macro_name, macro_value in macros.items():
            content = re.sub(r"(" + macro_name + r")" + r"([^a-zA-Z0-9])", macro_value+r"\2", content)

        if content == '':
            return []

        page[CONTENT] = content
        return [page]

    return modify

@factory_function
def arxiv_section_strip_modifier() -> List[Dict]:
    """
    Modify the input JSON object by removing any content before the first occurrence of a LaTeX section command
    or other section-like headers (e.g., chapter, subsection). This aims to strip the preamble or any content
    that precedes the main body of a LaTeX document.

    Taken from rpj_v1: https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/arxiv_cleaner.py

    Args:
        page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
                     that contains the text to be analyzed.

    Returns:
        List[Dict]: A list containing the input JSON object with content before the first section-like header removed.
                    If no section-like header is found, the CONTENT field is set to an empty string.
    """
    
    pattern = r"^(.*?)("
    pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r")"
    pattern = re.compile(pattern, flags=re.DOTALL)
    
    def modify(page):
        if not pattern.search(page[CONTENT]):
            return []

        new_content = pattern.sub(r"\2", page[CONTENT])
        if new_content == '':
            return []

        page[CONTENT] = new_content
        return [page]

    return modify
  
 
def stackexchange_list_modifier(page: Dict) -> List[Dict]:
    """
    Modifies the input JSON object for stackexchange pages by replacing HTML list tags with newline and bullet point symbols. 
    Specifically, this function replaces <li> tags with "\n*" and <ol> tags with "\n" to standardize list elements.

    It derives from the stackexchange processing pipeline from rpj_v1 and therefore operates on the page's 'question' and 
    'answers' keys instead of CONTENT. 

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a 'question' field and optionally a 'answers' field
                 that contains the text to be analyzed.

    Returns:
    List[Dict]: A list containing the input JSON object with HTML list tags replaced for 'question' and 'answers'
    """

    reformat_lists = lambda x: x.replace("<li>", "\n*").replace("</li>", "").replace("<ol>", "\n*").replace("</ol>", "")
    page['question']['text'] = reformat_lists(page['question']['text'])

    if 'answers' in page:
        for i, a in enumerate(page['answers']):
            page['answers'][i]['text'] = reformat_lists(a['text'])

    return [page]

def stackexchange_answer_sort_modifier(page: Dict, descending=True) -> List[Dict]:
    """
    Modifies the input JSON object for stackexchange pages by sorting the answers by their score metadata.

    It derives from the stackexchange processing pipeline from rpj_v1 and therefore operates on the page's 'question' and 
    'answers' keys instead of CONTENT. If 'answers' is not present, it is a no-op.

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a 'question' field and optionally a 'answers' field
                 that contains the text to be analyzed.
    descending (bool): When True, puts the highest scoring answers first. When False, puts lowest first.

    Returns:
    List[Dict]: A list containing the input JSON object with 'answers' re-ordered.
    """

    if 'answers' in page and isinstance(page['answers'], list):
        page['answers'].sort(key=lambda x: x.get('score', 0), reverse=descending)

    return [page]


@factory_function
def stackexchange_html_extraction_modifier():
    """
    Modifies the input JSON object for stackexchange pages by extracting the html for questions and answers. Uses BeautifulSoup
    as per the implementation in rpj_v1.  

    It derives from the stackexchange processing pipeline from rpj_v1 and therefore operates on the page's 'question' and 
    'answers' keys instead of CONTENT.

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a 'question' field and optionally a 'answers' field
                 that contains the text to be analyzed.

    Returns:
    List[Dict]: A list containing the input JSON object with 'questions' and answers' extracted.
    """

    extract = lambda h: BeautifulSoup(h, "lxml").get_text()
        
    def modify(page):
        page['question']['text'] = extract(page['question']['text'])
        if 'answers' in page:
            for i, a in enumerate(page['answers']):
                page['answers'][i]['text'] = extract(a['text'])
        return [page]

    return modify
      

def stackexchange_qa_formatter(page: Dict, remove_qa=False) -> List[Dict]:
    """
    Modifies the input JSON object for stackexchange pages by combining the 'question' and 'answers' fields into one string.
    Specifically, this uses the format 

    Q: Question text
    A: Answer 1 text
    A: Answer 2 text
    ⋮
    A: Answer N text

    It derives from the stackexchange processing pipeline from rpj_v1 and therefore operates on the page's 'question' and 
    'answers' keys to produce a unified string to be stored in CONTENT. By default the original 'quesiton' and 'answers'
    fields are kepy but they can be removed via the remove_qa arugment. 

    Args:
    page (Dict): A dictionary representing a JSON object. It should have a 'question' field and optionally a 'answers' field
                 that contains the text to be combined.
    remove_qa (Bool): Whether to remove the original "question" and "answers" fields

    Returns:
    List[Dict]: A list containing the input JSON object with 'questions' and answers' combined into a CONTENT field.
    """

    if 'question' not in page:
        return []

    if 'answers' in page:
        answers = "\nA: ".join([a["text"] for a in page["answers"]])
        page[CONTENT] = f"Q: {page['question']['text']}\nA: {answers}"
    else:
        page[CONTENT] = f"Q: {page['question']['text']}"

    if remove_qa:
        page.pop('question', None)
        page.pop('answers', None)

    return [page]

def move_url_modifier(page: Dict) -> List[Dict]:
    page[URL] = page['metadata']['WARC-Target-URI']
    return [page]

def key_name_modifier(page: Dict, old_key='content', new_key='text', allow_overwrite=False) -> List[Dict]:
    """
    Changes the name of a key in a page dictionary. Primarily used for handling outdated raw sources 
    where the CONTENT key is "content" instead of "text." If old_key is not present, this function
    is a no-op. If new_key is already present, allow_overwrite must be set to True for the function to 
    overwrite the previous value in new_key. 

    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            containing the raw HTML data obtained from CC
    old_key -- The name of the existing key that should be renamed
    new_key -- The new key name to replace old_key
    allow_overwrite -- Whether to overwrite the existing value for new_key. 

    Returns:
    A list containing the input JSON object that replaces the html in CONTENT with extracted text
    """
    if old_key in page:
        if new_key not in page or allow_overwrite:
            page[new_key] = page.pop(old_key)
        else: 
            warnings.warn(f"{new_key} is already in the page but allow_overwrite is set to False.")

    return [page]


def html_content_extraction_modifier(page: Dict) -> List[Dict]:
    """
    Uses the `justext` package to replace the full HTML with extracted text.

    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            containing the raw HTML data obtained from CC

    Returns:
    A list containing the input JSON object that replaces the html in CONTENT with extracted text
    """
    try:
        paragraphs = justext.justext(page[CONTENT], ())
    except ParserError:
        return []

    if paragraphs:
        page[CONTENT] = "\n\n".join(p.text for p in paragraphs)
        return [page]
    return []


@factory_function
def substring_line_modifier(banlist: Union[str, List], case_sensitive=False,
                            location='any', max_length=None, remove_substring_only=False) -> List[Dict]:
    """
    Filters the input JSON object - Remove lines that contain the given substring

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'content' field
            that contains the text to be analyzed.
    banlist -- The list of substrings that is banned
    case_sensitive -- Whether to ignore case when checking for banlist items in specific lines
    location -- Where the substring exists in the line. Options are {prefix, suffix, any}
    max_length -- Optional argument that specifies a maximum length of any removed line (that contains said substring)
    remove_substring_only -- Only remove the substring instead of removing the whole line

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """
    assert location in {'prefix', 'suffix', 'any'}

    if isinstance(banlist, str):
        banlist = [banlist]
    banlist = banlist if case_sensitive else [b.lower() for b in banlist]

    pattern = f"(?:{'|'.join(banlist)})"
    if location == 'prefix':
        pattern = rf"^{pattern}\s?"
    elif location == 'suffix':
        pattern = rf"\s?{pattern}$"
    else:
        pattern = rf"\s?{pattern}"

    pattern = re.compile(pattern) if case_sensitive else re.compile(pattern, re.I)

    def modify(page: Dict) -> List[Dict]:
        lines = page[CONTENT].split('\n')
        lines_without_substring = []
        for line in lines:
            if max_length is None or len(line.split()) <= max_length:
                if remove_substring_only:
                    modified_line = pattern.sub("", line)
                    if line and (not modified_line or modified_line.isspace()):
                        continue
                    line = modified_line
                elif pattern.search(line):
                    continue

            lines_without_substring.append(line)

        new_doc = '\n'.join(lines_without_substring).strip()

        if new_doc == '':
            return []

        page[CONTENT] = new_doc
        return [page]

    return modify


@factory_function
def punctuation_line_modifier(remove_ellipses=False):
    """
    Filters the input JSON object - Remove lines if they do not end in a punctuation mark

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'content' field
            that contains the text to be analyzed.
    remove_ellipses -- A boolean that specifies whether ellipses should count as punctuation. 
    To replicate the behavior of the C4 TFDS codebase, it shoud be set to True. 

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """
    pattern = re.compile(r'[.!?"\u2026]\s*$')
    ellipsis_pattern = re.compile(r'(?:\.\.\.|\u2026)\s*$')

    def modify(page: Dict) -> List[Dict]:
        lines = page[CONTENT].split('\n')
        # Create a new list that only includes lines that end in a punctuation mark.
        lines_with_punctuation = [line for line in lines if pattern.search(line) or not line]
        if remove_ellipses:
            lines_with_punctuation = [line for line in lines_with_punctuation if not ellipsis_pattern.search(line)]

        new_doc = '\n'.join(lines_with_punctuation)

        if new_doc == '':
            return []

        page[CONTENT] = new_doc
        return [page]

    return modify


def line_length_modifier(page: Dict, min_length=0, max_length=float('inf')) -> List[Dict]:
    """
    Filters the input JSON object - Remove lines with word counts outside accepted range
    (ps - Ideally, may want optional argument for prefix/suffix/substring banlist)

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'content' field
            that contains the text to be analyzed.
    min_length -- Minimum number of words to keep a line (inclusive).
    max_length -- Maximum number of words allowed to keep a line (inclusive).

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """
    lines = page[CONTENT].split('\n')
    lines_within_range = [line for line in lines if min_length <= len(
        line.split()) <= max_length or not line]  # TODO: Use proper word counts from a library?
    new_doc = '\n'.join(lines_within_range)

    if new_doc == '':
        return []

    page[CONTENT] = new_doc
    return [page]


def word_length_modifier(page: Dict, max_length=1000, **kwargs) -> List[Dict]:
    """
    Filters the input JSON object - Remove lines where the word with the largest length goes
    strictly over max_length. 

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'content' field
            that contains the text to be analyzed.
    max_length -- Maximum allowed length of a particular word. 
    kwargs -- kwargs for split_words
    
    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """
    lines = page[CONTENT].split('\n')
    lines_within_range = []
    for line in lines:
        words = split_words(line, **kwargs)
        if all(len(word) <= max_length for word in words):
            lines_within_range.append(line)

    new_doc = '\n'.join(lines_within_range)

    if new_doc == '':
        return []

    page[CONTENT] = new_doc
    return [page]


def uppercase_ratio_line_modifier(page: Dict, max_ratio=0.5) -> List[Dict]:
    """
    Filters the input JSON object - Remove lines where uppercase characers exceed a certain ratio

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'text' field
            that contains the text to be analyzed.
    max_ratio -- The maximum allowed ratio of uppercase characters

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """
    lines = page[CONTENT].split('\n')
    lines_below_ratio = []
    for line in lines:
        num_uppercase = sum(char.isupper() for char in line)
        if not line or num_uppercase / len(line) <= max_ratio:
            lines_below_ratio.append(line)

    new_doc = '\n'.join(lines_below_ratio).strip()

    if new_doc == '':
        return []

    page[CONTENT] = new_doc
    return [page]


def numeric_ratio_line_modifier(page: Dict, max_ratio=1.0) -> List[Dict]:
    """
    Filters the input JSON object - Remove lines if numerical characters exceed a certain ratio
    (ps - Falcon removes lines which contain 100% numerical characters)
    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    max_ratio -- The maximum allowed ratio of numeric characters. Note that max_ratio is not
            a strict threshold. To replicate the RefinedWeb rule which checks for all characters
            being numeric, set max_ratio as something like 1 - eps for eps = 1e-6.

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """
    lines = page[CONTENT].split('\n')
    lines_below_ratio = []
    for line in lines:
        num_numeric = sum(char.isdigit() for char in line)
        if not line or num_numeric / len(line) <= max_ratio:
            lines_below_ratio.append(line)

    new_doc = '\n'.join(lines_below_ratio).strip()

    if new_doc == '':
        return []

    page[CONTENT] = new_doc
    return [page]


@factory_function
def citation_removal_modifier() -> List[Dict]:
    """
    Modifies the input JSON object - Remove text related to citations (Wiki-format)

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'content' field
            that contains the text to be analyzed.

    Returns:
    A list containing the input JSON object with citation-related text removed.
    """

    citation_regex = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")

    def modify(page: Dict) -> List[Dict]:
        new_doc = citation_regex.sub("", page[CONTENT])

        if new_doc == '':
            return []

        page[CONTENT] = new_doc
        return [page]

    return modify


@factory_function
def url_removal_modifier(tlds_filepath="baselines/mappers/iana_tlds.txt"):
    """
    Modifies the input JSON object - Removes all urls within the content of a page, relying
    on two regexes for finding URLs: one relies upon a "vocab list" of existing top-level domains (TLDs),
    such as ".com", ".org", etc.; the other detects IP addresses

    Arguments:
    page -- A dictionary representing a JSON object. It should have a 'content' field
            that contains the text to be analyzed.
    tlds_filepath -- Path to a text file where the TLDs vocab is stored. The default is the path
            to the full list from IANA (https://www.iana.org/domains/root/db) assuming you are in the
            root directory of the dcnlp project.

    Returns:
    A list containing the input JSON object with urls in the text removec
    """
    with open(tlds_filepath, "r") as file:
        tlds_list = [re.escape(tld) for tld in file.read().splitlines()]

    # Create a simplified pattern to check if any TLDs are in the text
    tlds_regex = Blacklist(tlds_list, match_substrings=True).compiled

    # Detailed URL regex to detect URLs based on TLDs
    url_regex = re.compile(
        rf'\s{{0,10}}(?:((https?|ftp)://))?[-a-zA-Z0-9@:%._\+~#=]{{1,256}}\.({tlds_regex.pattern})\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    # Regex to detect IP addresses
    ipv4_regex = re.compile(
        r'\s{0,10}\b((https?|ftp)://)?(?:[0-2]?[0-9]{1,2}\.){3}[0-2]?[0-9]{1,2}[-a-zA-Z0-9()@:%_\+.~#?&//=]*')

    def modify(page: Dict) -> List[Dict]:
        # First, check for URLs based on TLDs
        if tlds_regex.match(page[CONTENT]):
            page[CONTENT] = url_regex.sub("", page[CONTENT])

        # Continue with removing IP addresses
        page[CONTENT] = ipv4_regex.sub("", page[CONTENT])

        if page[CONTENT] == '':
            return []

        return [page]

    return modify


@factory_function
def counter_line_modifier() -> List[Dict]:
    """
    Filters the input JSON object - Remove lines if it is a counter (e.g. 3 likes)
    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """

    # TODO: May want to add to this list ourselves or check with RefinedWeb folks to see what they did
    counter_regex = re.compile(
        r'^\W*\d(?:,|\.|\d)*(?:K|k|M|m|B|b)?\s+(?:likes|shares|comments|retweets|reposts|quotes|bookmarks|upvotes|downvotes|downloads|views|followers)\W*$'
    )

    def modify(page: Dict) -> List[Dict]:
        lines = page[CONTENT].split('\n')
        lines_without_counters = [line for line in lines if not counter_regex.search(line.lower())]
        new_doc = '\n'.join(lines_without_counters).strip()

        if new_doc == '':
            return []

        page[CONTENT] = new_doc
        return [page]

    return modify


def within_page_dedup(page: Dict, granularity: str = 'line', normalize=True, **kwargs) -> List[Dict]:
    """
    Modifies the input JSON object by removing exactly repeated text

    This function looks for exact duplicates according to a specified atomic unit. Right now this suppoers "line" and "paragraph"

    Only the first occurrence of a repeated unit of text is kept. 

    Arguments:
    page -- A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    granularity -- An string or int for how the repetition is measured. Accepted otions are {"line", "paragraph"} 
    normalize -- Whether or not to normalize the text within each unit (i.e., convert to lowercase and strip whitespace)
    kwargs -- Any parameters specific to one of the split_ functions in core_utils

    Returns:
    A list containing the input JSON object if it passes the filter,
    or an empty list if it doesn't.
    """

    if granularity == "line":
        split_text = split_paragraphs(page[CONTENT], paragraph_end='\n', remove_empty=False, **kwargs)
        sep = '\n'
    elif granularity == "paragraph":
        split_text = split_paragraphs(page[CONTENT], paragraph_end='\n\n', remove_empty=False, **kwargs)
        sep = '\n\n'

    seen_text = set()
    deduped_text = []

    for text in split_text:
        normalized_text = text.strip().lower() if normalize else text
        if normalized_text not in seen_text or not normalized_text:
            deduped_text.append(text)
            seen_text.add(normalized_text)

    page[CONTENT] = sep.join(deduped_text)
    return [page]


@factory_function
def newline_removal_modifier(max_consecutive=2):
    """
    This modifier normalizes line spacing by controlling for the maximum allowed consecutive newline characters ('\n')
    within a page.

    Arguments:
    - page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    - max_consecutive (int): The maximum number of consecutive newline characters to allow.

    Returns:
    A list containing the modified version of the input JSON object. 
    """
    pattern = re.compile(r'\n{%d,}' % (max_consecutive + 1))

    def modify(page):
        page[CONTENT] = pattern.sub('\n' * max_consecutive, page[CONTENT])
        return [page]

    return modify

def split_lines_modifier(page, delimiter='\n'):
    """
    This modifier splits the content of a page into a list of list of lines based on a delimiter
    If the page is empty, it instead gets removed. 
    Arguments:
    - page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    - delimiter (int): The character or substring to split on
    Returns:
    A list containing the modified version of the input JSON object. 
    """
    content = page[CONTENT]
    if isinstance(content, list): 
        return [page]
    elif isinstance(content, str):
        if len(content) == 0:
            return []
        else:
            page[CONTENT] = content.split(delimiter)
            return [page]
    else:
        raise TypeError


def join_lines_modifier(page, delimiter='\n'):
    """
    This modifier joins the content of a page if it is in a list (such as the output from deduplication).
    The specific delimiter to join on may be specified. 
    Arguments:
    - page (Dict): A dictionary representing a JSON object. It should have a CONTENT field
            that contains the text to be analyzed.
    - delimiter (int): The character or substring to join on
    Returns:
    A list containing the modified version of the input JSON object. 
    """
    content = page[CONTENT]
    if isinstance(content, str): 
        return [page]
    elif isinstance(content, list):
        if len(content) == 0:
            return []
        else:
            page[CONTENT] = delimiter.join(content)
            return [page]
    else:
        raise TypeError
