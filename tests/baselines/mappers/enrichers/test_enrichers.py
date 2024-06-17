import pytest
from baselines.mappers.enrichers.enrichers import *


def test_line_counter():
    # Test case 1
    result = line_counter('Hello\n\nWorld and all\nand beyond')
    assert result == len(['Hello', 'World and all',
                          'and beyond']), f"Expected 3, but got {result}"

    # Test case 2
    result = line_counter('Hello\n\nWorld and all\nand beyond', remove_empty=False)
    assert result == len(['Hello', '', 'World and all',
                          'and beyond']), f"Expected 4, but got {result}"

    # Test case 3
    result = line_counter('Hello\n\nWorld and all\nand beyond', '\n\n')
    assert result == len(['Hello',
                          'World and all\nand beyond']), f"Expected 2, but got {result}"

    # Test with an empty string
    result = line_counter('')
    assert result == len([]), f"Expected 0, but got {result}"

    # Test with no paragraphs
    result = line_counter('Hello World')
    assert result == len(['Hello World']), f"Expected 1, but got {result}"


def test_line_counter_enricher():
    page = {CONTENT: 'Hello\n\nWorld and all\nand beyond'}
    result = line_counter_enricher(page)
    assert result[0]['num_lines'] == 3, f"Expected 3, but got {result[0]['num_sentences']}"

def test_word_counter_enricher():
	# Empty page
	page = {CONTENT: ''}
	result = word_counter_enricher(page)
	assert result[0]['word_count'] == 0, f"Expected 0, but got {result[0]['word_count']}"	

	# Several different tricky cases included in the shared example
	page = {CONTENT: "CameCaseWord Line 1\nLine 2!!! e.g.\t  U.S.A.  \n\n\nLine  ????    3\t"}
	result = word_counter_enricher(page.copy())
	assert result[0]['word_count'] == 9, f"Expected 9, but got {result[0]['word_count']}"	

	result = word_counter_enricher(page.copy(), model='uniseg')
	assert result[0]['word_count'] == 9, f"Expected 9, but got {result[0]['word_count']}"	

	# Setting ignore punctuation to False can produce non-intuitive word counts when using either model
	result = word_counter_enricher(page.copy(), ignore_punctuation=False)
	assert result[0]['word_count'] == 14, f"Expected 14, but got {result[0]['word_count']}"	

	result = word_counter_enricher(page.copy(), model='uniseg', ignore_punctuation=False)
	assert result[0]['word_count'] == 18, f"Expected 18, but got {result[0]['word_count']}"	

	# Setting ignore whitespace to False can produce non-intuitive word counts when using uniseg
	result = word_counter_enricher(page.copy(), ignore_whitespace=False)
	assert result[0]['word_count'] == 9, f"Expected 9, but got {result[0]['word_count']}"	

	result = word_counter_enricher(page.copy(), model='uniseg', ignore_whitespace=False)
	assert result[0]['word_count'] == 29, f"Expected 9, but got {result[0]['word_count']}"	


