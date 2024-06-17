from baselines.mappers.filters import *
from baselines.core.constants import CONTENT
import pytest
import os

from baselines.mappers.filters import *

def test_github_extension_filter():
    # Test case with an allowed file extension
    page = {'filename': 'script.py'}
    assert github_extension_filter(page) == [page]

    # Check for allowed extensions that do not contain a "."
    page = {'filename': 'Makefile'}
    assert github_extension_filter(page) == [page]
    page = {'filename': 'Dockerfile'}
    assert github_extension_filter(page) == [page]

    # Test case with a different key for filename
    page = {'file_name': 'script.py'}
    assert github_extension_filter(page, filename_key='file_name') == [page]

    # Test case with a disallowed file extension
    page = {'filename': 'document.txt'}
    assert github_extension_filter(page) == []

    # Test case with no file extension
    page = {'filename': 'README'}
    assert github_extension_filter(page) == []

    # Test case with no specified filename key
    page = {CONTENT: 'Some content'}
    assert github_extension_filter(page) == []

    # Test with custom set of allowed extensions
    custom_extensions = ['.txt', '.md']
    page = {'filename': 'document.txt'}
    assert github_extension_filter(page, allowed_extensions=custom_extensions) == [page]

    page = {'filename': 'script.py'}
    assert github_extension_filter(page, allowed_extensions=custom_extensions) == []

def test_line_length_filters():
    content_max_exceeded = 'a' * 1001 + '\n' + 'Short line'
    page_max_exceeded = {CONTENT: content_max_exceeded}
    assert line_length_filter(page_max_exceeded, max_length=1000, length_type='max') == []

    # average line length exceeding limit
    content_avg_exceeded = ('a' * 200 + '\n') * 5
    page_avg_exceeded = {CONTENT: content_avg_exceeded}
    assert line_length_filter(page_avg_exceeded, max_length=100, length_type='avg') == []

    content_within_limits = 'a' * 100 + '\n' + 'Short line'
    page_within_limits = {CONTENT: content_within_limits}
    assert line_length_filter(page_within_limits, max_length=100, length_type='avg') == [page_within_limits]
    assert line_length_filter(page_within_limits, max_length=1000, length_type='max') == [page_within_limits]

    # Single line files should be allowed
    content_single_line = 'a' * 100
    page_single_line = {CONTENT: content_single_line}
    assert line_length_filter(page_single_line) == [page_single_line]

    # Test case with average and maximum line lengths within limits
    content = "Short line\n" * 50  # Each line has length 11 including newline, total 550 characters over 50 lines
    page = {CONTENT: content}
    assert line_length_filter(page, max_length=11, length_type='max') == [page]
    assert line_length_filter(page, max_length=11, length_type='avg') == [page]

    # Test case with maximum line length exceeding limit
    content = "Short line\n" * 9 + "A" * 1001 + "\n"  # 9 short lines, one long line of 1001 characters
    page = {CONTENT: content}
    assert line_length_filter(page, max_length=1000, length_type='max') == []

    # Test case with both average and maximum line lengths exceeding limits
    content = "A" * 200 + "\n" + "B" * 800 + "\n" + "C" * 300  # Three lines with lengths 200, 800, and 300
    page = {CONTENT: content}
    assert line_length_filter(page, max_length=750, length_type='max') == []
    assert line_length_filter(page, max_length=150, length_type='avg') == []

    # Test case with empty content
    page = {CONTENT: ''}
    assert line_length_filter(page, length_type='max') == []
    assert line_length_filter(page, length_type='avg') == []

    # Test case with custom limits
    content = "Medium length line\n" * 20  # Each line has length 19 including newline, total 380 characters over 20 lines
    page = {CONTENT: content}
    assert line_length_filter(page, max_length=20, length_type='max') == [page]
    assert line_length_filter(page, max_length=20, length_type='avg') == [page]

def test_alphabetic_characters_to_tokens_filter():
    alpha_to_tokens_filter = alphabetic_characters_to_tokens_filter()

    # Test with empty text
    page = {CONTENT: ''}
    assert alpha_to_tokens_filter(page) == []

    # Test with all alphanumeric text and default max_alpha_ratio
    page = {CONTENT: 'Hello123'}
    assert alpha_to_tokens_filter(page) == [page]
    assert alpha_to_tokens_filter(page, 0.25) == [page]

    # Test with a text that has a low alphabetical ratio
    page = {CONTENT: '12345'}
    assert alpha_to_tokens_filter(page) == []

    # Test with a text that has a high alphabetical ratio
    page = {CONTENT: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    assert alpha_to_tokens_filter(page, 0.5) == [page]

    # Test with a text that has equal numbers of alphanumeric and non-alphanumeric characters
    page = {CONTENT: 'A1B2C3'}
    assert alpha_to_tokens_filter(page) == []

    # Test with a text that contains special characters
    page = {CONTENT: '!@#$%^&*()_+'}
    assert alpha_to_tokens_filter(page, 0.1) == []

    # Test with a text that contains spaces
    page = {CONTENT: 'Hello World'}
    assert alpha_to_tokens_filter(page, 0.5) == [page]

def test_alphanumeric_char_ratio_filter():
    # Test with empty text
    page = {CONTENT: ''}
    assert alphanumeric_char_ratio_filter(page) == []

    # Test with all alphanumeric text
    page = {CONTENT: 'Hello123'}
    assert alphanumeric_char_ratio_filter(page, 0.25) == [page]

    # Test with mixed content
    page = {CONTENT: 'Hello 123!'}
    assert alphanumeric_char_ratio_filter(page, 0.25) == [page]

    # Test with no alphanumeric characters
    page = {CONTENT: '!@#$%'}
    assert alphanumeric_char_ratio_filter(page, 0.25) == []

    # Boundary ratio test
    page = {CONTENT: 'abcd'}  # 100% alphanumeric
    assert alphanumeric_char_ratio_filter(page, 1.0) == [page]
    assert alphanumeric_char_ratio_filter(page, 0.0) == [page]

    # Edge cases with unusual characters and whitespace
    page = {CONTENT: 'ab cd\n123!@#'}
    assert alphanumeric_char_ratio_filter(page, 0.25) == [page]
    page = {CONTENT: '    '}
    assert alphanumeric_char_ratio_filter(page, 0.25) == []

def test_repetition_filter():
    page = {CONTENT: ''}
    assert repetition_filter(page, granularity='line', max_fraction=1) == []
    assert repetition_filter(page, granularity='paragraph', max_fraction=1) == [] 
    assert repetition_filter(page, granularity=2, max_fraction=0.99) == []
    assert repetition_filter(page, granularity=8, max_fraction=0.99) == []
    
    ### Line-wise repetition filtering ###
    
    # If just one line/paragraph, then return the page by default
    page = {CONTENT: 'Hello, World.'}
    assert repetition_filter(page, granularity='line', max_fraction=0.0, count_characters=False) == [page]
    assert repetition_filter(page, granularity='line', max_fraction=0.0, count_characters=True) == [page]

    # Sould ignore empty lines when multiple \n characters follow each other
    page = {CONTENT: 'Hello\nHello\n\n\n\nHello'}
    assert repetition_filter(page, granularity='line', max_fraction=1, count_characters=False) == [page]
    assert repetition_filter(page, granularity='line', max_fraction=0.99, count_characters=False) == []
    assert repetition_filter(page, granularity='line', max_fraction=1, count_characters=True) == [page]
    assert repetition_filter(page, granularity='line', max_fraction=0.99, count_characters=True) == []

    # Some repeated lines and some not
    page = {CONTENT: 'Hello, World.\nHello, World.\n\nHello'}
    assert repetition_filter(page, granularity='line', max_fraction=2/3+1e-2, count_characters=False) == [page]
    assert repetition_filter(page, granularity='line', max_fraction=2/3-1e-2, count_characters=False) == []
    assert repetition_filter(page, granularity='line', max_fraction=26/31+1e-2, count_characters=True) == [page]
    assert repetition_filter(page, granularity='line', max_fraction=26/31-1e-2, count_characters=True) == []
    
    ### Paragraph-level repetition filtering ###
    
    # Mix of paragrpahs and lines
    page = {CONTENT: 'Hello, World.\n\nHello, World.\n\nHello, World.\nHello'}
    assert repetition_filter(page, granularity='paragraph', max_fraction=2/3+1e-2, count_characters=False) == [page]
    assert repetition_filter(page, granularity='paragraph', max_fraction=2/3-1e-2, count_characters=False) == []
    assert repetition_filter(page, granularity='paragraph', max_fraction=26/45+1e-2, count_characters=True) == [page]
    assert repetition_filter(page, granularity='paragraph', max_fraction=26/45-1e-2, count_characters=True) == []

    ### N-gram-level repetition filtering ###

    # If no n-grams can be formed, then just return page
    page = {CONTENT: 'hello'}
    assert repetition_filter(page, granularity=2, max_fraction=0.0) == [page]
    assert repetition_filter(page, granularity=5, max_fraction=0.0) == [page] 

    # All characters are part of most repeated {2,3,4}-grams
    page = {CONTENT: 'hello world hello world hello world'}
    assert repetition_filter(page, granularity=2, max_fraction=0.99) == []
    assert repetition_filter(page, granularity=3, max_fraction=0.99) == []
    assert repetition_filter(page, granularity=4, max_fraction=0.99) == []

    # Should ignore punctuation and spacing
    page = {CONTENT: 'hello      world hello apple.... world hello\n world hello'}
    assert repetition_filter(page, granularity=2, max_fraction=30/40+1e-2) == [page]
    assert repetition_filter(page, granularity=2, max_fraction=30/40-1e-2) == []
    assert repetition_filter(page, granularity=3, max_fraction=30/40+1e-2) == [page]
    assert repetition_filter(page, granularity=3, max_fraction=30/40-1e-2) == []
    assert repetition_filter(page, granularity=4, max_fraction=0) == [page]

    # When ties exist for most common n_gram, use the longest one (in chars) to compare against the threshold
    page = {CONTENT: 'a b a b a b ab cd ab cd ab cd abc def abc def'}
    assert repetition_filter(page, granularity=2, max_fraction=12/30+1e-2) == [page]
    assert repetition_filter(page, granularity=2, max_fraction=12/30-1e-2) == []

    page = {CONTENT: 'a b a b a b abc def abc def ab cd ab cd ab cd'}
    assert repetition_filter(page, granularity=2, max_fraction=12/30+1e-2) == [page]
    assert repetition_filter(page, granularity=2, max_fraction=12/30-1e-2) == []

    page = {CONTENT: 'hello world hello\n world hello. apple world hello world hello world hello world'}
    assert repetition_filter(page, granularity=5, max_fraction=60/65+1e-2) == [page]
    assert repetition_filter(page, granularity=5, max_fraction=60/65-1e-2) == []

@pytest.mark.timeout(1)
def test_massive_web_repetition_filters():
    for _ in range(100):
        page = {CONTENT: 'This is a complex sentence with unique words.\n Here is another one.'}
        assert massive_web_repetition_filters(page) == [page]

        page = {CONTENT: 'Repeat repeat repeat repeat.'}
        assert massive_web_repetition_filters(page) == []

        page = {CONTENT: ''}
        assert massive_web_repetition_filters(page) == []

        page = {CONTENT: None}
        with pytest.raises(TypeError):
            massive_web_repetition_filters(page)

        page = {
            CONTENT: (
                'This is a complex sentence with unique words.\n'
                'Here is another one.\n We can say a lot of things, because we want this page to not be filtered out.\n'
                'However, in order to make sure all parts of the code are reached,\nwe craete a long n-gram that is repeated:\n'
                'a complex sentence with unique words.\nNow just making sure that the freq count is low enough so it will not filter out anything.\n'
                'Since our n-gram is quite long, need to add a lot of text here in order to verify it is indeed not filtered out. I hope that is enough.\n'
                'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. '
                'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.'
            )
        }
        assert massive_web_repetition_filters(page) == [page]

def test_page_length_filter():
    # Test with empty text
    page = {CONTENT: ''}
    assert page_length_filter(page, "word", 1) == []
    assert page_length_filter(page, "sentence", 1) == []
    assert page_length_filter(page, "line", 1) == []
    assert page_length_filter(page, "paragraph", 1) == []
    assert page_length_filter(page, "char", 1) == []
    
    # Test with a one-sentence long piece of text
    page = {CONTENT: 'Hello world.'}
    assert page_length_filter(page, "word", 3) == []
    assert page_length_filter(page, "word", 2) == [page]
    assert page_length_filter(page, "sentence", 2) == []
    assert page_length_filter(page, "sentence", 1) == [page]
    assert page_length_filter(page, "line", 2) == []
    assert page_length_filter(page, "line", 1) == [page]
    assert page_length_filter(page, "paragraph", 2) == []
    assert page_length_filter(page, "paragraph", 1) == [page]
    assert page_length_filter(page, "char", 13) == []
    assert page_length_filter(page, "char", 12) == [page]

    # Test with a multi-sentence piece of text
    page = {CONTENT: 'Hello world. This is a test.'}
    assert page_length_filter(page, "word", 7) == []
    assert page_length_filter(page, "word", 6) == [page]
    assert page_length_filter(page, "sentence", 3) == []
    assert page_length_filter(page, "sentence", 2) == [page]
    assert page_length_filter(page, "line", 2) == []
    assert page_length_filter(page, "line", 1) == [page]
    assert page_length_filter(page, "paragraph", 2) == []
    assert page_length_filter(page, "paragraph", 1) == [page]
    assert page_length_filter(page, "char", 29) == []
    assert page_length_filter(page, "char", 28) == [page]
    
    # Test with a multi-line piece of text
    page = {CONTENT: 'Hello world. \n This is a test.'}
    assert page_length_filter(page, "word", 7, ignore_punctuation=True) == []
    assert page_length_filter(page, "word", 6, ignore_punctuation=True) == [page]
    assert page_length_filter(page, "sentence", 3) == []
    assert page_length_filter(page, "sentence", 2) == [page]
    assert page_length_filter(page, "line", 3) == []
    assert page_length_filter(page, "line", 2) == [page]
    assert page_length_filter(page, "paragraph", 2) == []
    assert page_length_filter(page, "paragraph", 1) == [page]
    assert page_length_filter(page, "char", 31) == []
    assert page_length_filter(page, "char", 30) == [page]

    # Test a multi-paragraph piece of text
    page = {CONTENT: 'Hello world. \n\n This is a test.'}
    assert page_length_filter(page, "word", 7, ignore_punctuation=True) == []
    assert page_length_filter(page, "word", 6, ignore_punctuation=True) == [page]
    assert page_length_filter(page, "sentence", 3) == []
    assert page_length_filter(page, "sentence", 2) == [page]
    assert page_length_filter(page, "line", 3) == []
    assert page_length_filter(page, "line", 2) == [page]
    assert page_length_filter(page, "paragraph", 3) == []
    assert page_length_filter(page, "paragraph", 2) == [page]
    assert page_length_filter(page, "char", 32) == []
    assert page_length_filter(page, "char", 31) == [page]

    # Test robustness to when multiple \n characters appear in a row
    page = {CONTENT: 'Hello world. \n\n\n \n\n\n This is a test.'}
    assert page_length_filter(page, "word", 7, ignore_punctuation=True) == []
    assert page_length_filter(page, "word", 6, ignore_punctuation=True) == [page]
    assert page_length_filter(page, "sentence", 3) == []
    assert page_length_filter(page, "sentence", 2) == [page]
    assert page_length_filter(page, "line", 3) == []
    assert page_length_filter(page, "line", 2) == [page]
    assert page_length_filter(page, "paragraph", 3) == []
    assert page_length_filter(page, "paragraph", 2) == [page]
    assert page_length_filter(page, "char", 37) == []
    assert page_length_filter(page, "char", 36) == [page]

    # Test sentence counting when text with no period
    page = {CONTENT: 'Hello world This is a test Python is great'}
    assert page_length_filter(page, "sentence", 2) == []
    assert page_length_filter(page, "sentence", 1) == [page]

    # Test sentence counting when text has periods not representing end of sentences
    page = {CONTENT: 'Mr. Smith bought 2.5 kilograms of fruit.'}
    assert page_length_filter(page, "word", 8, ignore_punctuation=True) == []
    assert page_length_filter(page, "word", 7, ignore_punctuation=True) == [page]
    assert page_length_filter(page, "sentence", 2) == []
    assert page_length_filter(page, "sentence", 1) == [page]

    # Test sentence counting when some sentences have alternative punctuation
    page = {CONTENT: 'Hello world! That should be easy, right? I think so...yeah...'}
    assert page_length_filter(page, "word", 12, model="uniseg", ignore_punctuation=True) == []      
    assert page_length_filter(page, "word", 11, model="uniseg", ignore_punctuation=True) == [page]  # This requires uniseg to pass. fasttext treats so...yeah... as one word  
    assert page_length_filter(page, "sentence", 4) == []
    assert page_length_filter(page, "sentence", 3) == [page]

    # Test with None as text
    page = {CONTENT: None}
    with pytest.raises(TypeError):
        page_length_filter(page, "word", 1)

    # Test with no CONTENT in page
    page = {}
    with pytest.raises(KeyError):
        page_length_filter(page, "word", 1)

    # Test with negative minimum sentence count
    page = {CONTENT: 'Hello.'}
    assert page_length_filter(page, "word", -1) == [page]


def test_substring_filter():
    # Test basic presence/absence of the banned word
    page = {CONTENT: "warning: Javascript should be enabled"}
    assert substring_filter('javascript')(page) == []
    assert substring_filter('{')(page) == [page]

    # Test when multiple banned words are passed in
    assert substring_filter(['javascript', '{'])(page) == []
    assert substring_filter(['python', '{'])(page) == [page]

    # Test location parameter
    assert substring_filter('javascript', location='prefix')(page) == [page]
    assert substring_filter('javascript', location='suffix')(page) == [page]
    assert substring_filter('warning', location='prefix')(page) == []
    assert substring_filter(['javascript', 'warning'], location='prefix')(page) == []
    assert substring_filter('enabled', location='suffix')(page) == []
    assert substring_filter(['javascript', 'enabled'], location='suffix')(page) == []

    # Test case sensitivity
    assert substring_filter('javascript', case_sensitive=True)(page) == [page]
    assert substring_filter('Javascript', case_sensitive=True)(page) == []

    # Test exact word detection
    page = {CONTENT: "warning: Javascript should be enabled"}
    assert substring_filter('java', exact_word=True)(page) == [page]
    assert substring_filter('javascript', exact_word=True)(page) == []
    assert substring_filter('warning', location='prefix', exact_word=True)(page) == []
    assert substring_filter('warn', location='prefix', exact_word=True)(page) == [page]
    assert substring_filter('enabled', location='suffix', exact_word=True)(page) == []
    assert substring_filter('abled', location='suffix', exact_word=True)(page) == [page]

    # Test loading words from file
    fname = 'tests/baselines/mappers/filters/banlists/ldnoobw.txt' if os.getcwd().endswith("/dcnlp") else "banlists/ldnoobw.txt"
    ldnoobw_filter = substring_filter(banlist_from_fname=fname, exact_word=False) # Instantiating explicitly to test factory_function
    ldnoobw_filter_exact_word = substring_filter(banlist_from_fname=fname, exact_word=True)
    
    page = {CONTENT: "this text contains a badword xxx so it should be removed"}
    assert ldnoobw_filter(page) == []
    assert ldnoobw_filter_exact_word(page) == []
    page = {CONTENT: "this text contains a badword xxx123 so it should be removed"}
    assert ldnoobw_filter(page) == []
    assert ldnoobw_filter_exact_word(page) == [page]
    page = {CONTENT: "this text contains a badword 123xxx so it should be removed"}
    assert ldnoobw_filter(page) == []
    assert ldnoobw_filter_exact_word(page) == [page]
    page = {CONTENT: "this text does not contain a bad word so it should not be removed"}
    assert ldnoobw_filter(page) == [page]
    assert ldnoobw_filter_exact_word(page) == [page]

    # Test with empty page
    page = {}
    with pytest.raises(KeyError):
        substring_filter('javascript')(page)


def test_ellipsis_count_filter():
    for ell in ['...', '. . .', '\u2026']:

        # Text with 50% of lines ending with an ellipsis
        page = {CONTENT: f"This text has one line that ends with an ellipsis{ell}\n and one line that doesn't."}
        assert ellipsis_count_filter(page, max_ellipsis_end_ratio=0.49) == []
        assert ellipsis_count_filter(page, max_ellipsis_end_ratio=0.51) == [page]

        # Text where ellipsis exists but not at end of a line
        page = {CONTENT: f"This text has one{ell}line that ends with an ellipsis{ell}and one line that doesn't."}
        assert ellipsis_count_filter(page, max_ellipsis_end_ratio=0.01) == [page]

        # Text where multiple ellipsis exist in a row
        page = {CONTENT: f"This text has one{ell}{ell}\n{ell}line that ends with an ellipsis{ell}\nand one line that doesn't."}
        assert ellipsis_count_filter(page, max_ellipsis_end_ratio=0.66) == []
        assert ellipsis_count_filter(page, max_ellipsis_end_ratio=0.67) == [page]


def test_bullet_count_filter():
    for bullet1 in ['●', '•', '*', '-']:
        for bullet2 in ['●', '•', '*', '-']:

            # Text with 2/3 of lines start with an ellipsis, using both acceptable forms of ellipsis
            page = {CONTENT: f"{bullet1} This text has two lines\n{bullet2} that start with a bullet\n and one line that doesn't."}
            assert bullet_count_filter(page, max_bullet_start_ratio=0.66) == []
            assert bullet_count_filter(page, max_bullet_start_ratio=0.67) == [page]

            # Text where bullets exist but not at end of a line
            page = {CONTENT: f"This text has two lines{bullet2} that start with a bullet{bullet1}\n and one line that doesn't."}
            assert bullet_count_filter(page, max_bullet_start_ratio=0.01) == [page]

            # Text where multiple bullets exist in a row
            page = {CONTENT: f"{bullet1}{bullet1}{bullet1}{bullet1}{bullet1}This text has two lines\n{bullet2} that start with a bullet\n and one line that doesn't."}
            assert bullet_count_filter(page, max_bullet_start_ratio=0.66) == []
            assert bullet_count_filter(page, max_bullet_start_ratio=0.67) == [page]


def test_stop_word_filter():
    # Differentiate instances of stop words vs unique stop words
    page = {CONTENT: 'The best of the best'} # the, of, the
    assert stop_word_filter(page, False, min_stop_word=3) == [page]
    assert stop_word_filter(page, True, min_stop_word=3) == []

    # All stop words
    page = {CONTENT: 'the be to of and that have with'}
    assert stop_word_filter(page, False, min_stop_word=8) == [page]
    assert stop_word_filter(page, True, min_stop_word=8) == [page]

    # Substrings
    page = {CONTENT: 'Then Tom and Sally withheld bees'} # and
    assert stop_word_filter(page, False, min_stop_word=2) == []
    assert stop_word_filter(page, True, min_stop_word=2) == []

    # Require 'text' field
    page = {}
    with pytest.raises(KeyError):
        stop_word_filter(page, False, min_stop_word=3)
    with pytest.raises(KeyError):
        stop_word_filter(page, True, min_stop_word=3)


def test_word_length_filter():
    # Regular case
    page = {CONTENT: 'The average word length is 4.33'}
    assert word_length_filter(page, min_length=4, max_length=5) == [page]

    # Test short words and default value
    page = {CONTENT: 'The cat in the hat'}
    assert word_length_filter(page, min_length=3, max_length=10) == []
    assert word_length_filter(page, max_length=10) == [page]

    # Test long words and default value
    page = {CONTENT: 'Loquaciousness verylongwords'}
    assert word_length_filter(page, min_length=3, max_length=10) == []
    assert word_length_filter(page, min_length=3) == [page]

    # Test no words
    page = {CONTENT: '  '}
    assert word_length_filter(page, min_length=3, max_length=10) == []

    # Require 'text' field
    page = {}
    with pytest.raises(KeyError):
        word_length_filter(page, min_length=4, max_length=5)


def test_symbol_ratio_filter():
    # Regular tests
    page = {CONTENT: 'No symbols here.'}
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=0.0) == [page]
    page = {CONTENT: '#hashtags #on #every #word'}
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=1.0) == [page]
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=0.1) == []
    # Test all ellipses
    page = {CONTENT: '#hashtags and ellipses both count...'}
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=0.3) == []
    page = {CONTENT: '#hashtags and ellipses both count\u2026'}
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=0.3) == []
    page = {CONTENT: '#hashtags and ellipses both count. . .'}
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=0.25) == []

    # No words
    page = {CONTENT: '   '}
    assert symbol_ratio_filter(page, max_symbol_to_word_ratio=0.1) == []

    # Missing 'text'
    page = {}
    with pytest.raises(KeyError):
        symbol_ratio_filter(page, max_symbol_to_word_ratio=0.1)

        
def test_word_removal_ratio_filter():
    # Scenario 1: No words are removed, should return the page as is.
    page = {CONTENT: "word 1\nword 2\nword 3", 'prev_word_count': 6}
    assert word_removal_ratio_filter(page, 'prev_word_count') == [page]

    # Scenario 2: 33.33% of words are removed (2 out of 6), which is more than 5%.
    page = {CONTENT: "word 1\nword 2", 'prev_word_count': 6}
    assert word_removal_ratio_filter(page, 'prev_word_count') == []
    assert word_removal_ratio_filter(page, 'prev_word_count', max_removed_ratio=0.34) == [page]

    # Scenario 3: Exactly 5% of words are removed (95 remain out of 100). Should return the page as is.
    page = {CONTENT: "word\n" * 95, 'prev_word_count': 100}
    assert word_removal_ratio_filter(page, 'prev_word_count') == [page]

    # Scenario 4: Just above 5% of words are removed (94 remain out of 100). Should return [].
    page = {CONTENT: "word\n" * 94, 'prev_word_count': 100}
    assert word_removal_ratio_filter(page, 'prev_word_count') == []

    # Scenario 5: No words in the original or modified text. Should return [].
    page = {CONTENT: "", 'prev_word_count': 0}
    assert word_removal_ratio_filter(page, 'prev_word_count') == []

    # Scenario 6: 100% of words are removed (0 remain out of 100). Should return [].
    page = {CONTENT: "", 'prev_word_count': 100}
    assert word_removal_ratio_filter(page, 'prev_word_count') == []

    # Scenario 7: Ignores superfluous whitespace and punctuation. Setting ignore_punctuation=False can produce weird counts
    page = {CONTENT: " Line 1\nLine 2!!!\t    \n\n\nLine  ????    3\t", 'prev_word_count': 12}
    assert word_removal_ratio_filter(page, 'prev_word_count', max_removed_ratio=0.5) == [page]
    page = {CONTENT: " Line 1\nLine 2!!!\t    \n\n\nLine  ????    3\t", 'prev_word_count': 13}
    assert word_removal_ratio_filter(page, 'prev_word_count', max_removed_ratio=0.5) == []
    assert word_removal_ratio_filter(page, 'prev_word_count', max_removed_ratio=0.5, ignore_punctuation=False) == [page]

    # Scenario 9: "new_word_count_key" exists and is used even if it is incorrect
    page = {CONTENT: "Line 1\nLine 2\nLine 3", 'prev_word_count': 6, 'new_word_count': 6}
    assert word_removal_ratio_filter(page, 'prev_word_count', 'new_word_count') == [page]

    page = {CONTENT: "Line 1\nLine 2\nLine 3", 'prev_word_count': 99, 'new_word_count': 6}
    assert word_removal_ratio_filter(page, 'prev_word_count', 'new_word_count') == []
    
    page = {CONTENT: "Line 1\nLine 2\nLine 3", 'prev_word_count': 99, 'new_word_count': 99}
    assert word_removal_ratio_filter(page, 'prev_word_count', 'new_word_count') == [page]
    
 
def test_alphabetic_word_ratio_filter():
    # Test with empty page
    page = {CONTENT: ''}
    assert alphabetic_word_ratio_filter(page, 0.2) == []

    # Test with 20% non-alphabetic words and some words with a mix of alphabetic and non-alphabetic elements
    page = {CONTENT: 'This sentence contains <1,000 non-alphabetic w0rd5 within it but >1.'}
    assert alphabetic_word_ratio_filter(page, 0.2) == [page]
    assert alphabetic_word_ratio_filter(page, 0.1) == []

    # Test with 20% non-alphabetic words and some words with a mix of alphabetic and non-alphabetic elements
    page = {CONTENT: 'This sentence contains <1,000     non-alphabetic \nw0rd5 within it but >1.'}
    assert alphabetic_word_ratio_filter(page, 0.2) == [page]
    assert alphabetic_word_ratio_filter(page, 0.1) == []
