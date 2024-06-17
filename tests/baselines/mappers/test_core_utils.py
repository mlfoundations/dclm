import pytest
from baselines.mappers.core_utils import *


def test_split_paragraphs():
    # Test case 1
    result = split_paragraphs('Hello\n\nWorld and all\nand beyond')
    assert result == ['Hello', 'World and all',
                      'and beyond'], f"Expected ['Hello', 'World and all', 'and beyond'], but got {result}"

    # Test case 2
    result = split_paragraphs('Hello\n\nWorld and all\nand beyond', remove_empty=False)
    assert result == ['Hello', '', 'World and all',
                      'and beyond'], f"Expected ['Hello', '', 'World and all', 'and beyond'], but got {result}"

    # Test case 3
    result = split_paragraphs('Hello\n\nWorld and all\nand beyond', '\n\n')
    assert result == ['Hello',
                      'World and all\nand beyond'], f"Expected ['Hello', 'World and all\nand beyond'], but got {result}"

    # Test with an empty string
    result = split_paragraphs('')
    assert result == [], f"Expected [], but got {result}"

    # Test with no paragraphs
    result = split_paragraphs('Hello World')
    assert result == ['Hello World'], f"Expected ['Hello World'], but got {result}"


def test_split_sentences():
    # Test case 1
    result = split_sentences(
        'Hello. World and all. And beyond. A sentence with an abbreviation,'
        ' e.g. a sentence. Also one with a 1.4 floating points?')
    expected_result = ['Hello.', 'World and all.', 'And beyond.', 'A sentence with an abbreviation, e.g. a sentence.',
                       'Also one with a 1.4 floating points?']
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

    # Additional test cases:
    # Test with an empty string
    result = split_sentences('')
    assert result == [], f"Expected [], but got {result}"

    # Test with no periods
    result = split_sentences('Hello World')
    assert result == ['Hello World'], f"Expected ['Hello World'], but got {result}"

    # Test with tokenizer='nltk'
    result = split_sentences('Hello. World and all.', tokenizer='nltk')
    assert result == ['Hello.', 'World and all.'], f"Expected ['Hello.', 'World and all.'], but got {result}"

    result = split_sentences('This we I sit down with Wildman Weiler and K-Love (new to the show) and we talk as the title says, simply drunken... This was a random off the wall show... Enjoy!', tokenizer='nltk')
    assert result == ['This we I sit down with Wildman Weiler and K-Love (new to the show) and we talk as the title says, simply drunken... This was a random off the wall show... Enjoy!'], f"Expected ['This we I sit down with Wildman Weiler and K-Love (new to the show) and we talk as the title says, simply drunken... This was a random off the wall show... Enjoy!'], but got {result}"

    result = split_sentences('Hello. World and all.', tokenizer='nltk', tokenizer_lang='en')
    assert result == ['Hello.', 'World and all.'], f"Expected ['Hello.', 'World and all.'], but got {result}"    

    result = split_sentences('This we I sit down with Wildman Weiler and K-Love (new to the show) and we talk as the title says, simply drunken... This was a random off the wall show... Enjoy!', tokenizer='nltk', tokenizer_lang='en')
    assert result == ['This we I sit down with Wildman Weiler and K-Love (new to the show) and we talk as the title says, simply drunken...', 'This was a random off the wall show...', 'Enjoy!'], f"Expected ['This we I sit down with Wildman Weiler and K-Love (new to the show) and we talk as the title says, simply drunken...', 'This was a random off the wall show...', 'Enjoy!'], but got {result}"

    # Test with invalid tokenizer
    with pytest.raises(ValueError):
        split_sentences('Hello. World and all.', tokenizer='invalid')


def test_split_words():
    # Test case 1
    result = len(split_words('Hello World and all', model='uniseg', ignore_punctuation=True))
    assert result == 4, f"Expected 4, but got {result}"

    # Additional test cases:
    # Test with different punctuation and whitespace settings
    result = len(split_words('Also one with complications e.g., a 1.4 floating points...', model='uniseg',
                             ignore_punctuation=True))
    assert result == 9, f"Expected 9, but got {result}"

    result = len(split_words('Also one with complications e.g., a 1.4 floating points...', model='uniseg',
                             ignore_punctuation=False))
    assert result == 14, f"Expected 14, but got {result}"

    result = len(split_words('Also one with complications e.g., a 1.4 floating points...', model='uniseg',
                             ignore_whitespace=False, ignore_punctuation=True))
    assert result == 17, f"Expected 17, but got {result}"

    # Test with different models
    result = len(split_words('One with CamelCasedWords', model='uniseg'))
    assert result == 3, f"Expected 3, but got {result}"

    result = len(split_words('Hello World and all', model='fasttext', ignore_punctuation=True))
    assert result == 4, f"Expected 4, but got {result}"

    result = len(split_words('Hello World and all', model='split', ignore_punctuation=True))
    assert result == 4, f"Expected 4, but got {result}"

    # Test with invalid model
    with pytest.raises(ValueError):
        split_words('Hello World and all', model='invalid')


def test_join_sentences():
    # Test case 1
    result = join_sentences(['Hello', 'World and all', 'and beyond'])
    assert result == 'Hello World and all and beyond', f"Expected 'Hello World and all and beyond', but got {result}"

    # Additional test cases:
    # Test with an empty list
    result = join_sentences([])
    assert result == '', f"Expected '', but got {result}"

    # Test with a list of one sentence
    result = join_sentences(['Hello'])
    assert result == 'Hello', f"Expected 'Hello', but got {result}"

    # Test with a custom separator
    result = join_sentences(['Hello', 'World'], sep='-')
    assert result == 'Hello-World', f"Expected 'Hello-World', but got {result}"


def test_join_paragraphs():
    # Test case 1
    result = join_paragraphs(['Hello', 'World and all', 'and beyond'])
    expected = 'Hello\nWorld and all\nand beyond'
    assert result == expected, f"Expected {expected}, but got {result}"

    # Additional test cases:
    # Test with an empty list
    result = join_paragraphs([])
    assert result == '', f"Expected '', but got {result}"

    # Test with a list of one paragraph
    result = join_paragraphs(['Hello'])
    assert result == 'Hello', f"Expected 'Hello', but got {result}"

    # Test with a custom separator
    result = join_paragraphs(['Hello', 'World'], sep='-')
    assert result == 'Hello-World', f"Expected 'Hello-World', but got {result}"

def test_normalize_url():
    # Reference URL for testing first regex rule r"https?:\/\/(www\.)?"
    url_1 = 'https://www.google.com'
    normalized_url_1 = normalize_url(url_1)

    # http instead of https
    url_2 = 'http://www.google.com'
    assert normalized_url_1 == normalize_url(url_2)

    # http instead of https
    url_2 = 'google.com'
    assert normalized_url_1 == normalize_url(url_2)

    # www need not be present
    url_2 = 'https://google.com'
    assert normalized_url_1 == normalize_url(url_2)

    # different domain
    url_2 = 'https://www.yahoo.com'
    assert normalized_url_1 != normalize_url(url_2)

    # Reference URL 2 for testing second regex rule r"\?(utm_|ref|feed).*"
    url_1 = 'https://jobs.shoppersdrugmart.ca/job/?feedId=4&utm_source=Indeed'
    normalized_url_1 = normalize_url(url_1)

    # nothing in place of "?feedID=...""
    url_2 = 'https://jobs.shoppersdrugmart.ca/job/'
    assert normalized_url_1 == normalize_url(url_2)

    # removal of "?ref*" 
    url_2 = 'https://jobs.shoppersdrugmart.ca/job/?ref=123'
    assert normalized_url_1 == normalize_url(url_2)

    # removal of "?utm_*"
    url_2 = 'http://jobs.shoppersdrugmart.ca/job/?utm_source=Indeed'
    assert normalized_url_1 == normalize_url(url_2)

    # test in conjunction with first rule
    url_2 = 'http://www.jobs.shoppersdrugmart.ca/job/'
    assert normalized_url_1 == normalize_url(url_2)

    # different domain
    url_2 = 'https://jobs.google.com/job/?feedId=4&utm_source=Indeed'
    assert normalized_url_1 != normalize_url(url_2)


def test_normalize_whitespace_and_lowercase():
    # Reference text
    text_1 = 'This is a piece of sample text.  '
    normalized_text_1 = normalize_whitespace_and_lowercase(text_1)

    # All lowercase
    text_2 = 'this is a piece of sample text.'
    assert normalized_text_1 == normalize_whitespace_and_lowercase(text_2)

    # All uppercase
    text_2 = 'THIS IS A PIECE OF SAMPLE TEXT.'
    assert normalized_text_1 == normalize_whitespace_and_lowercase(text_2)

    # Leading whitespace
    text_2 = '\n This is a piece of sample text.'
    assert normalized_text_1 == normalize_whitespace_and_lowercase(text_2)

    # Trailing whitespace
    text_2 = 'This is a piece of sample text. \t'
    assert normalized_text_1 == normalize_whitespace_and_lowercase(text_2)

    # Difference in case and whitespace
    text_2 = 'This is a PIECE OF SAMPLE TEXT.   '
    assert normalized_text_1 == normalize_whitespace_and_lowercase(text_2)

    # Same case by different words
    text_2 = 'This is a difference piece of text.'
    assert normalized_text_1 != normalize_whitespace_and_lowercase(text_2)


def test_normalize_timestamps():
    # Reference timestamp
    timestamp_1 = '2013-12-05T04:56:51Z'
    normalized_timestamp_1 = normalize_timestamps(timestamp_1)

    # Difference is in the second
    timestamp_2 = '2013-12-05T04:56:50Z'
    assert normalized_timestamp_1 > normalize_timestamps(timestamp_2)

    # Difference is in the minute
    timestamp_2 = '2013-12-05T04:55:51Z'
    assert normalized_timestamp_1 > normalize_timestamps(timestamp_2)

    # Difference is in the hour
    timestamp_2 = '2013-12-05T03:56:51Z'
    assert normalized_timestamp_1 > normalize_timestamps(timestamp_2)

    # Difference is in the day
    timestamp_2 = '2013-12-04T04:56:51Z'
    assert normalized_timestamp_1 > normalize_timestamps(timestamp_2)

    # Difference is in the month
    timestamp_2 = '2013-11-05T04:56:51Z'
    assert normalized_timestamp_1 > normalize_timestamps(timestamp_2)

    # Difference is in the year
    timestamp_2 = '2012-11-05T04:56:51Z'
    assert normalized_timestamp_1 > normalize_timestamps(timestamp_2)

    # Improperly formatted timestamp
    assert normalize_timestamps("time") == -1 

def test_ccnet_dedup_normalizer():
    weird = "０２３´∶：\x10 | ;012 hèllo"
    normalized = "000 | ;000 hello"
    assert normalized == ccnet_dedup_normalizer(weird)

    weird = "😃 kožušček"
    normalized = " kozuscek"
    assert normalized == ccnet_dedup_normalizer(weird)

    weird = "北亰 29 \xa0"
    normalized = "Bei Jing  00"
    assert normalized == ccnet_dedup_normalizer(weird)