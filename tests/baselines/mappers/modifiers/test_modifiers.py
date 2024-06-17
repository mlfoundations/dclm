import pytest
import warnings
import copy

from baselines.mappers.modifiers import *


def test_key_name_modifier():
    # Test when the old_key is present
    page = {'content': 'example text'}
    assert key_name_modifier(page, 'content', 'text') == [{'text': 'example text'}]
    
    # If old_key not already present, don't do anything. 
    page = {'content': 'example text'}
    assert key_name_modifier(page, 'page_content', 'text') == [{'content': 'example text'}]

    # If old_key and new_key are present, do not overwrite new_key by default
    page = {'content': 'example text', 'text': 'existing text'}
    with pytest.warns(UserWarning, match="text is already in the page but allow_overwrite is set to False."):
        assert key_name_modifier(page, 'content', 'text') == [{'content': 'example text', 'text': 'existing text'}]
    assert key_name_modifier(page, 'content', 'text', allow_overwrite=True) == [{'text': 'example text'}]

    
def test_starcoder_v2_repo_splitter():
    input_page = {
        'repo_name': 'sample/repo', 
        'files': [ {'filename': f'/path/to/file{i}', 'text': f'code{i}'} for i in range(30)],
        CONTENT: 'placeholder text'
    }

    output_pages = [
        {
            'repo_name': 'sample/repo', 
            'files': [ {'filename': f'/path/to/file{i}', 'text': f'code{i}'} for i in range(10)]
        },
        {
            'repo_name': 'sample/repo', 
            'files': [ {'filename': f'/path/to/file{i}', 'text': f'code{i}'} for i in range(10,20,1)]
        },
        {
            'repo_name': 'sample/repo', 
            'files': [ {'filename': f'/path/to/file{i}', 'text': f'code{i}'} for i in range(20, 30, 1)]
        }
    ]

    assert starcoder_v2_repo_splitter(input_page, max_files=10) == output_pages
    assert starcoder_v2_repo_splitter(input_page, max_files=30) == [input_page]

    
def test_starcoder_v2_format_modifier():
    input_page = {
        'repo_name': 'sample/repo', 
        'files': [
            {'filename': '/path/to/file1', 'text': 'code1'},
            {'filename': '/path/to/file2', 'text': 'code2'},
            {'filename': '/path/to/file3', 'text': 'code3'},
        ]
    }

    output_page = copy.deepcopy(input_page)

    text_w_metadata = "<repo_name>sample/repo<file_sep>/path/to/file1\ncode1<file_sep>/path/to/file2\ncode2<file_sep>/path/to/file3\ncode3"
    text_wo_metadata = "<file_sep>code1<file_sep>code2<file_sep>code3"
    text_w_metadata_wo_delim = "sample/repo\n\n/path/to/file1\ncode1\n\n/path/to/file2\ncode2\n\n/path/to/file3\ncode3"
    text_wo_metadata_wo_delim = "code1\n\ncode2\n\ncode3"

    page = copy.deepcopy(input_page)
    output_page[CONTENT] = text_w_metadata
    assert starcoder_v2_format_modifier(page, add_metadata_prob=1) == [output_page]

    page = copy.deepcopy(input_page)
    output_page[CONTENT] = text_wo_metadata
    assert starcoder_v2_format_modifier(page, add_metadata_prob=0) == [output_page]

    page = copy.deepcopy(input_page)
    output_page[CONTENT] = text_w_metadata_wo_delim
    assert starcoder_v2_format_modifier(page, add_metadata_prob=1, add_sentinels=False) == [output_page]

    page = copy.deepcopy(input_page)
    output_page[CONTENT] = text_wo_metadata_wo_delim
    assert starcoder_v2_format_modifier(page, add_metadata_prob=0, add_sentinels=False) == [output_page]
    

# Test cases for arxiv_appendix_modifier
def test_arxiv_appendix_modifier():
    modifier = arxiv_appendix_modifier()

    # Test with \appendix at the beginning
    input_page = {CONTENT: "\\appendix\nAppendix"}
    assert modifier(input_page) == []

    # Test with \bibliography at the beginning
    input_page = {CONTENT: "\\bibliography{ref}\nBibliography content"}
    assert modifier(input_page) == []

    # Test with variations of \begin{references}
    input_page = {CONTENT: "Content\n\\begin{references}\nReferences content"}
    expected_output = {CONTENT: "Content\n"}
    assert modifier(input_page) == [expected_output]

    input_page = {CONTENT: "Content\n\\begin{REFERENCES}\nREFERENCES content"}
    expected_output = {CONTENT: "Content\n"}
    assert modifier(input_page) == [expected_output]

    input_page = {CONTENT: "Content\n\\begin{thebibliography}\nBibliography content"}
    expected_output = {CONTENT: "Content\n"}
    assert modifier(input_page) == [expected_output]

    # Test with content before and after the specified headers
    input_page = {CONTENT: "Main content\n\\appendix\nAppendix content\n\\bibliography{ref}\nBibliography content"}
    expected_output = {CONTENT: "Main content\n"}
    assert modifier(input_page) == [expected_output]

def test_arxiv_comment_modifier():

    modifier = arxiv_comment_modifier()
    modifier_w_multiline = arxiv_comment_modifier(remove_multiline=True)

    # Test removing a single-line comment
    input_page = {CONTENT: "% This is a single line comment"}
    assert modifier(input_page) == []

    # Test removing multiple single-line comments
    input_page = {CONTENT: "% Comment 1\n% Comment 2\nContent"}
    expected_output = {CONTENT: "Content"}
    assert modifier(input_page) == [expected_output]

    # Test removing mixed single and multi-line comments
    input_page = {CONTENT: "Content before % Comment\n\\begin{comment}Multi-line comment\\end{comment}Content after"}
    expected_output = {CONTENT: "Content before\nContent after"}
    assert modifier_w_multiline(input_page) == [expected_output]

    input_page = {CONTENT: "Content before % Comment\n\\begin{comment}Multi-line comment\\end{comment}Content after"}
    expected_output = {CONTENT: "Content before\n\\begin{comment}Multi-line comment\\end{comment}Content after"}
    assert modifier(input_page) == [expected_output]

    # Test removing comments with content in between
    input_page = {CONTENT: "% This is a single line comment\nSome content here\n\\begin{comment}This is a multiline comment\\end{comment}More content"}
    expected_output = {CONTENT: "Some content here\nMore content"}
    assert modifier_w_multiline(input_page) == [expected_output]

    input_page = {CONTENT: "% This is a single line comment\nSome content here\n\\begin{comment}This is a multiline comment\\end{comment}More content"}
    expected_output = {CONTENT: "Some content here\n\\begin{comment}This is a multiline comment\\end{comment}More content"}
    assert modifier(input_page) == [expected_output]

    # Test with content after a single line comment
    input_page = {CONTENT: "Content before comment % This is a comment\nContinued content"}
    expected_output = {CONTENT: "Content before comment\nContinued content"}
    assert modifier(input_page) == [expected_output]

    # Test with empty content
    input_page = {CONTENT: ""}
    assert modifier(input_page) == []

    # Test with no comments
    input_page = {CONTENT: "Regular content without comments"}
    expected_output = {CONTENT: "Regular content without comments"}
    assert modifier(input_page) == [expected_output]

    # Additional test case for in-line comments
    input_page = {CONTENT: "Content before inline % comment\nMore content"}
    expected_output = {CONTENT: "Content before inline\nMore content"}
    assert modifier(input_page) == [expected_output]

def test_arxiv_section_strip_modifier():
    modifier = arxiv_section_strip_modifier()

    # Test with content before and after the first section
    input_page = {CONTENT: "Content before\\section{First Section}Content in between\\section{Last Section}Content after"}
    expected_output = {CONTENT: "\\section{First Section}Content in between\\section{Last Section}Content after"}
    assert modifier(input_page) == [expected_output]

    # Test with multiple sections and preamble
    input_page = {CONTENT: "Preamble\\section{Sec1}Content1\\section{Sec2}Content2"}
    expected_output = {CONTENT: "\\section{Sec1}Content1\\section{Sec2}Content2"}
    assert modifier(input_page) == [expected_output]

    # Test with only one section
    input_page = {CONTENT: "Preamble\\section{OnlySection}Content"}
    expected_output = {CONTENT: "\\section{OnlySection}Content"}
    assert modifier(input_page) == [expected_output]

    # Test with no sections
    input_page = {CONTENT: "Content without any section"}
    assert modifier(input_page) == []

    # Test with empty content
    input_page = {CONTENT: ""}
    assert modifier(input_page) == []

def test_arxiv_macro_modifier():
    modifier = arxiv_macro_modifier()

    # Test replacing a \newcommand macro with its value
    input_page = {CONTENT: "\\newcommand{\\examplemacro}{ExampleValue}\nUse of \\examplemacro here."}
    expected_output = {CONTENT: '\\newcommand{ExampleValue}{ExampleValue}\nUse of ExampleValue here.'} 
    assert modifier(input_page) == [expected_output]

    # Test replacing multiple macros
    input_page = {CONTENT: "\\newcommand{\\macroOne}{FirstValue}\n\\newcommand{\\macroTwo}{SecondValue}\nUse of \\macroOne and \\macroTwo."}
    expected_output = {CONTENT: "\\newcommand{FirstValue}{FirstValue}\n\\newcommand{SecondValue}{SecondValue}\nUse of FirstValue and SecondValue."}
    assert modifier(input_page) == [expected_output]

    # Test with content but no macros
    input_page = {CONTENT: "Regular LaTeX content without macros"}
    expected_output = {CONTENT: "Regular LaTeX content without macros"}
    assert modifier(input_page) == [expected_output]

    
def test_stackexchange_list_modifier():
    page_wo_answers = {'question': {'text': 'Question 1\n<ol>\n\t<li>Choice 1</li>\n\t<li>Choice 2</li></ol>'}}
    stackexchange_list_modifier(page_wo_answers) == [{'question': {'text': "Question 1\n\n*\n\t\n*Choice 1\n\t\n*Choice 2"}}]

    page = {
        'question': {'text': 'Question 1\n<ol>\n\t<li>Choice 1</li>\n\t<li>Choice 2</li></ol>'},
        'answers': [
            {'text': 'Choice 1'},
            {'text': '<ol>\n\t<li>Choice 1 is correct</li>\n\t<li>Choice 2 is incorrect</li></ol>'},
        ]
    }

    output_page = {
        'question': {'text': 'Question 1\n\n*\n\t\n*Choice 1\n\t\n*Choice 2'},
        'answers': [
            {'text': 'Choice 1'},
            {'text': '\n*\n\t\n*Choice 1 is correct\n\t\n*Choice 2 is incorrect'},
        ]
    }

    assert stackexchange_list_modifier(page) == [output_page]


def test_stackexchange_answer_sort_modifier():
    page_with_scores = {
        'question': {'text': 'Question 1'},
        'answers': [
            {'text': 'Answer 1', 'score': 10},
            {'text': 'Answer 2', 'score': 5},
            {'text': 'Answer 3', 'score': 15},
        ]
    }

    sorted_page_descending = {
        'question': {'text': 'Question 1'},
        'answers': [
            {'text': 'Answer 3', 'score': 15},
            {'text': 'Answer 1', 'score': 10},
            {'text': 'Answer 2', 'score': 5},
        ]
    }

    sorted_page_ascending = {
        'question': {'text': 'Question 1'},
        'answers': [
            {'text': 'Answer 2', 'score': 5},
            {'text': 'Answer 1', 'score': 10},
            {'text': 'Answer 3', 'score': 15},
        ]
    }

    assert stackexchange_answer_sort_modifier(page_with_scores) == [sorted_page_descending]
    assert stackexchange_answer_sort_modifier(page_with_scores, descending=False) == [sorted_page_ascending]

    page_wo_answers = {'question': {'text': 'Question 1'}}

    assert stackexchange_answer_sort_modifier(page_wo_answers) == [{'question': {'text': 'Question 1'}}]


def test_stackexchange_html_extraction_modifier():
    modifier = stackexchange_html_extraction_modifier()

    page = {
        'question': {'text': '<!DOCTYPE html>\n<html>\n<head><title>Question title</title></head>\n<body>Question body</body>\n</html>'},
        'answers': [
            {'text': 'Answer 3', 'score': 15},
            {'text': '<ol>\n\t<li>Here is the answer.</li>\n\t<li>This is why.</li></ol>', 'score': 10},
        ]
    }

    output_page = {
        'question': {'text': '\nQuestion title\nQuestion body\n'},
        'answers': [
            {'text': 'Answer 3', 'score': 15},
            {'text': '\nHere is the answer.\nThis is why.', 'score': 10},
        ]
    }

    assert modifier(page) == [output_page]

    page_wo_answers = {
        'question': {'text': '<!DOCTYPE html>\n<html>\n<head><title>Question title</title></head>\n<body>Question body</body>\n</html>'},
    }

    assert modifier(page_wo_answers) == [{'question': {'text': '\nQuestion title\nQuestion body\n'}}]
    

def test_stackexchange_qa_formatter():
    page = {
        'question': {'text': 'Question 1'},
        'answers': [
            {'text': 'Answer 3', 'score': 15},
            {'text': 'Answer 1', 'score': 10},
            {'text': 'Answer 2', 'score': 5},
        ]
    }

    output_page = {
        'question': {'text': 'Question 1'},
        'answers': [
            {'text': 'Answer 3', 'score': 15},
            {'text': 'Answer 1', 'score': 10},
            {'text': 'Answer 2', 'score': 5},
        ],
        CONTENT: "Q: Question 1\nA: Answer 3\nA: Answer 1\nA: Answer 2"
    }

    output_page_wo_qa = {CONTENT: "Q: Question 1\nA: Answer 3\nA: Answer 1\nA: Answer 2"}

    assert stackexchange_qa_formatter(page.copy()) == [output_page]
    assert stackexchange_qa_formatter(page.copy(), remove_qa=True) == [output_page_wo_qa]

    page_wo_answers = {'question': {'text': 'Question 1'}}

    assert stackexchange_qa_formatter(page_wo_answers.copy()) == [{'question': {'text': 'Question 1'}, CONTENT: "Q: Question 1"}]
    assert stackexchange_qa_formatter(page_wo_answers.copy(), remove_qa=True) == [{ CONTENT: "Q: Question 1"}]

    page_wo_question = {
        'answers': [
            {'text': 'Answer 3', 'score': 15},
            {'text': 'Answer 1', 'score': 10},
            {'text': 'Answer 2', 'score': 5},
        ]
    }

    assert stackexchange_qa_formatter(page_wo_question) == []


def test_html_content_extraction_modifier():
    # Test with empty text
    page = {CONTENT: ''}
    assert html_content_extraction_modifier(page) == []

    # Test with short 404 page
    page = {
        CONTENT: '<html><head><meta charset="utf-8" /><title>404 Not Found</title><meta name="description" content="404 Not Found" /></head></html>'}
    assert html_content_extraction_modifier(page) == []

    # Test with short sample page
    page = {
        CONTENT: '<!DOCTYPE html>\n<html>\n<head><title>Sample title</title></head>\n<body>Sample body</body>\n</html>'}
    assert html_content_extraction_modifier(page) == [{
        CONTENT: 'Sample body'  # replaced the original content with the text extracted from the HTML
    }]


def test_substring_line_modifier():
    # Test with no CONTENT in page
    page = {}
    with pytest.raises(KeyError):
        substring_line_modifier('javascript')(page)

    # Test with empty text
    page = {CONTENT: ''}
    assert substring_line_modifier('javascript')(page) == []

    # Test with various lines
    page = {CONTENT: 'JavaScript must be enabled\n\nThis is an article about JavaScript.\njava script enabled '}
    assert substring_line_modifier('javascript')(page) == [{CONTENT: 'java script enabled'}]

    # Test with case sensitivity
    page = {CONTENT: 'JavaScript must be enabled\n\nThis is an article about JavaScript.\njava script enabled '}
    assert substring_line_modifier('javascript', case_sensitive=True)(page) == [
        {CONTENT: 'JavaScript must be enabled\n\nThis is an article about JavaScript.\njava script enabled'}]

    # Test with different substring
    page = {CONTENT: 'JavaScript must be enabled\nThis is an article about JavaScript.\n\njava script enabled '}
    assert substring_line_modifier('enabled')(page) == [{CONTENT: 'This is an article about JavaScript.'}]

    # Test with banlist with multiple items
    page = {CONTENT: 'JavaScript must be enabled\nThis is an article about JavaScript.\n\njava script enabled '}
    assert substring_line_modifier(['javascript', 'enabled'])(page) == []

    # Test with non-trivial max_length argument (to reproduce RefinedWeb rule)
    page = {
        CONTENT: 'You have 7 items in cart right now\nYou have 7 items in cart right now but you can add more by clicking back onto the homepage.'}
    assert substring_line_modifier('items in cart', location='any', max_length=10)(page) == [
        {CONTENT: 'You have 7 items in cart right now but you can add more by clicking back onto the homepage.'}]
    page = {
        CONTENT: 'You have 7 items in cart right now\n  items in cart  \nYou have 7 items in cart right now but you can add more by clicking back onto the homepage.'}
    assert substring_line_modifier('items in cart', location='any', max_length=10, remove_substring_only=True)(
        page) == [{
        CONTENT: 'You have 7 right now\nYou have 7 items in cart right now but you can add more by clicking back onto the homepage.'}]

    # Test with non-trivial location and non-trivial max_length argument (to reproduce RefinedWeb rules)
    page = {
        CONTENT: 'sign-in for full access\nThis is an article about how to sign-in.\n\nsign-in at this page by providing your username and password. If you forget your password you may contact a site administrator.'}
    assert substring_line_modifier('sign-in', location='prefix', max_length=10)(page) == [{
        CONTENT: 'This is an article about how to sign-in.\n\nsign-in at this page by providing your username and password. If you forget your password you may contact a site administrator.'}]
    page = {
        CONTENT: 'sign-in for full access\nsign-in  \nThis is an article about how to sign-in.\n\nsign-in at this page by providing your username and password. If you forget your password you may contact a site administrator.'}
    assert substring_line_modifier('sign-in', location='prefix', max_length=10, remove_substring_only=True)(page) == [{
        CONTENT: 'for full access\nThis is an article about how to sign-in.\n\nsign-in at this page by providing your username and password. If you forget your password you may contact a site administrator.'}]

    page = {
        CONTENT: 'Please read more...\nRead more...and find out!\n\nThis is a short summary about the article, which can be expanded upon if you wish. Read more...'}
    assert substring_line_modifier('Read more...', location='suffix', max_length=10)(page) == [{
        CONTENT: 'Read more...and find out!\n\nThis is a short summary about the article, which can be expanded upon if you wish. Read more...'}]
    page = {
        CONTENT: 'Please read more...\n  read more...\nRead more...and find out!\n\nThis is a short summary about the article, which can be expanded upon if you wish. Read more...'}
    assert substring_line_modifier('Read more...', location='suffix', max_length=10, remove_substring_only=True)(
        page) == [{
        CONTENT: 'Please\nRead more...and find out!\n\nThis is a short summary about the article, which can be expanded upon if you wish. Read more...'}]


def test_line_length_modifier():
    # Test with empty text
    page = {CONTENT: ''}
    assert line_length_modifier(page, 2, 10) == []

    # Test with various lines
    page = {CONTENT: 'Short line\n\na b c d e f g h i j k\n  There are exactly 10 words present in this line here\nHi'}
    assert line_length_modifier(page, 2, 10) == [
        {CONTENT: 'Short line\n\n  There are exactly 10 words present in this line here'}]

    # Test with unspecified max
    page = {CONTENT: 'Short line\n\na b c d e f g h i j k\n  There are exactly 10 words present in this line here\nHi'}
    assert line_length_modifier(page, min_length=2) == [
        {CONTENT: 'Short line\n\na b c d e f g h i j k\n  There are exactly 10 words present in this line here'}]

    # Test with unspecified min
    page = {CONTENT: 'Short line\n\na b c d e f g h i j k\n  There are exactly 10 words present in this line here\nHi'}
    assert line_length_modifier(page, max_length=10) == [
        {CONTENT: 'Short line\n\n  There are exactly 10 words present in this line here\nHi'}]

    # Test with no CONTENT in page
    page = {}
    with pytest.raises(KeyError):
        line_length_modifier(page, 2, 10)


def test_punctuation_line_modifier():
    # punctuation_line_modifier is a factory function
    modifier = punctuation_line_modifier()
    # Test with empty text
    page = {CONTENT: ''}
    assert modifier(page) == []

    # Test with various lines
    page = {
        CONTENT: 'Period.\nQuestion?\nExclamation!\n\n"Quotation"\nNothing\nWhitespace.  \nEllipsis...\nEllipsis\u2026'}
    assert modifier(page) == [{
        CONTENT: 'Period.\nQuestion?\nExclamation!\n\n"Quotation"\nWhitespace.  \nEllipsis...\nEllipsis\u2026'}]

    # Test with ellipses
    modifier = punctuation_line_modifier(remove_ellipses=True)
    page = {
        CONTENT: 'Period.\nQuestion?\nExclamation!\n\n"Quotation"\nEllipsis... \nNothing\nWhitespace.  \nEllipsis\u2026'}
    assert modifier(page) == [{CONTENT: 'Period.\nQuestion?\nExclamation!\n\n"Quotation"\nWhitespace.  '}]

    # Test with no CONTENT in page
    page = {}
    with pytest.raises(KeyError):
        modifier(page)


def test_word_length_modifier():
    # Test with empty text
    page = {CONTENT: ''}
    assert word_length_modifier(page, 10, model='split') == []
    # Test with various lines
    page = {CONTENT: 'Regular line    \n\nVerylongword and regular words\nBaaaaarely in the limit.'}
    assert word_length_modifier(page, 10, ignore_punctuation=True, model='split') == [
        {CONTENT: 'Regular line    \n\nBaaaaarely in the limit.'}]
    # Test with no 'text' in page
    page = {}
    with pytest.raises(KeyError):
        word_length_modifier(page, 10, model='split')


@pytest.mark.timeout(10)
def test_counter_line_modifier():
    counter_remover = counter_line_modifier()

    # Test with empty text
    page = {CONTENT: ''}
    assert counter_remover(page) == []

    # Test with various different counters and number types
    for counter_type in ["likes", "shares", "comments", "retweets", "reposts", "quotes", "bookmarks", "upvotes",
                         "downvotes", "downloads", "views", "followers"]:
        page = {
            CONTENT: f'3 {counter_type}\n3,000 {counter_type}\n123K {counter_type}!\n 5.1M   {counter_type}  \nThis line does not contain a counter.'}
        assert counter_remover(page) == [{CONTENT: "This line does not contain a counter."}]

    # Test whether a line that contains a counter only as a substring is kept in 
    page = {CONTENT: '123K quotes\n\nHello, World.\n\nHello, World with 3 views.\n3 views'}
    assert counter_remover(page) == [{CONTENT: "Hello, World.\n\nHello, World with 3 views."}]

    # Test that regex does not hang on long numbers
    page = {CONTENT: '12345' * 10 + '\n' + '12345' * 10 + ' likes'}
    counter_remover(page)


def test_uppercase_ratio_line_modifier():
    # Test with empty text
    page = {CONTENT: ''}
    assert uppercase_ratio_line_modifier(page) == []

    # Test single line case
    page = {CONTENT: 'Hello, World'}
    assert uppercase_ratio_line_modifier(page, 1.0) == [{CONTENT: 'Hello, World'}]
    assert uppercase_ratio_line_modifier(page, 2 / 12) == [{CONTENT: 'Hello, World'}]
    assert uppercase_ratio_line_modifier(page, 2 / 12 - 1e-6) == []
    assert uppercase_ratio_line_modifier(page, 0.0) == []

    # Test whether it removes only the incorrect line and handles extra line breaks
    page = {CONTENT: 'HELLO WORLD\n\nhello world\nHello, World\n\nHello, world'}
    assert uppercase_ratio_line_modifier(page, 1.0) == [
        {CONTENT: 'HELLO WORLD\n\nhello world\nHello, World\n\nHello, world'}]
    assert uppercase_ratio_line_modifier(page, 2 / 12) == [{CONTENT: 'hello world\nHello, World\n\nHello, world'}]
    assert uppercase_ratio_line_modifier(page, 1 / 12) == [{CONTENT: 'hello world\n\nHello, world'}]
    assert uppercase_ratio_line_modifier(page, 0) == [{CONTENT: 'hello world'}]


def test_numeric_ratio_line_modifier():
    # Test with empty text
    page = {CONTENT: ''}
    assert numeric_ratio_line_modifier(page) == []

    # Test with line of all numbers (as per the RefinedWeb rule)
    page = {CONTENT: '12345'}
    assert numeric_ratio_line_modifier(page, 1.0) == [{CONTENT: '12345'}]
    assert numeric_ratio_line_modifier(page, 1.0 - 1e-6) == []
    assert numeric_ratio_line_modifier(page, 0.0) == []

    # Test whether it removes only the incorrect line and handles extra line breaks
    page = {CONTENT: '1a\n\n2bb\n\n3cc\nabcd'}
    assert numeric_ratio_line_modifier(page, 1.0) == [{CONTENT: '1a\n\n2bb\n\n3cc\nabcd'}]
    assert numeric_ratio_line_modifier(page, 0.5) == [{CONTENT: '1a\n\n2bb\n\n3cc\nabcd'}]
    assert numeric_ratio_line_modifier(page, 1 / 3) == [{CONTENT: '2bb\n\n3cc\nabcd'}]
    assert numeric_ratio_line_modifier(page, 0.0) == [{CONTENT: 'abcd'}]


def test_citation_removal_modifier():
    # Test with empty text
    page = {CONTENT: ''}
    assert citation_removal_modifier()(page) == []

    # Test with various lines
    page = {
        CONTENT: '[edit]This is what a wiki citation looks like[134].\nStudies show it helps to remove it for LLM training[citation needed].'}
    assert citation_removal_modifier()(page) == [
        {CONTENT: 'This is what a wiki citation looks like.\nStudies show it helps to remove it for LLM training.'}]

    # Test with no 'text' in page
    page = {}
    with pytest.raises(KeyError):
        citation_removal_modifier()(page)


def test_url_removal_modifier():
    url_remover = url_removal_modifier()

    # Test with empty text
    page = {CONTENT: ''}
    assert url_remover(page) == []

    # Test with various url formats (generated by chatGPT)
    test_urls = [
        'https://www.example.com',
        'www.example.org',
        'example.co.uk'
        'example.randomsite.net:9000',
        'blog.samplewebsite.org:8081/post.html',
        'ftp://ftp.example.com/files',
        'ftp://files.example.net:21/downloads',
        'irc.randomchat.org:6667/chatroom',
        'mail.samplecompany.com:25/inbox',
        'www.technewsportal.org:8084/latest-news',
        'gamingcommunity.com:8085/forums/7890',
        'https://blog.samplewebsite.org',
        'https://en.wikipedia.org/wiki/Random_topic',
        'http://www.ecommerce-store.biz/products/sale',
        'https://www.example.com/search?q=url',
        'https://www.example.com/page?id=123&category=tech',
        'https://www.example.com#section1',
        'https://www.example.com/page#comments',
        'ftp://username:password@ftp.example.com',
        'http://username:password@example.com',
        'http://www.example.museum',
        'http://bit.ly/abcdef',
        'https://tinyurl.com/xyz123',
        'http://192.168.1.1',
        'https://10.0.0.1:8080',
        '1.1.1.1'
    ]

    for url in test_urls:
        page = {CONTENT: f'This URL {url} should be removed.'}
        assert url_remover(page) == [{CONTENT: 'This URL should be removed.'}]

        # When URL appears on its own line
        page = {CONTENT: f'This URL should be removed\n{url}\nThis text should still be on a new line.'}
        assert url_remover(page) == [{CONTENT: f'This URL should be removed\nThis text should still be on a new line.'}]

    # Test with common occurrences which are NOT URLs
    test_not_urls = [
        'U.S.A.',
        'e.g.',
        '3.14',
        '1.123.456.7899'  # Phone number formatted with dots
        '999.999.999.999'  # Invalid IPv4 Address
        '123.123.123'
        'v3.12.7.4'
    ]
    for string in test_not_urls:
        page = {CONTENT: f'This text {string} is not a URL.\n{string}'}
        assert url_remover(page) == [{CONTENT: f'This text {string} is not a URL.\n{string}'}]

    # Test with no 'text' in page
    page = {}
    with pytest.raises(KeyError):
        url_remover(page)


def test_within_page_dedup():
    # Line-level: check whether it correctly ignores whitespace and case
    page = {
        CONTENT: 'This line is duplicated.\n\nThis line is duplicated.\n This line is duplicated.  \n\nthis line is duplicated.\n This line is kept.'}
    assert within_page_dedup(page, 'line') == [{CONTENT: 'This line is duplicated.\n\n\n This line is kept.'}]

    # Line-level: check whether it checks for whitespace and case
    page = {
        CONTENT: 'This line is duplicated.\nThis line is duplicated.\n This line is duplicated.  \n\nthis line is duplicated.\nThis line is kept.'}
    assert within_page_dedup(page, 'line', normalize=False) == [{
        CONTENT: 'This line is duplicated.\n This line is duplicated.  \n\nthis line is duplicated.\nThis line is kept.'}]

    # Paragraph-level: check whether it ignores whitespace and case
    page = {
        CONTENT: 'This line is duplicated.\nThis line is duplicated.\n\n this line is duplicated.\nthis line is duplicated.\n\nThis line is duplicated.'}
    assert within_page_dedup(page, 'paragraph') == [
        {CONTENT: 'This line is duplicated.\nThis line is duplicated.\n\nThis line is duplicated.'}]

    # Paragraph-level: check whether it checks for whitespace and case
    page = {
        CONTENT: 'This line is duplicated.\nThis line is duplicated.\n\n this line is duplicated.\nthis line is duplicated.\n\nThis line is duplicated.\nThis line is duplicated.'}
    assert within_page_dedup(page, 'paragraph', normalize=False) == [{
        CONTENT: 'This line is duplicated.\nThis line is duplicated.\n\n this line is duplicated.\nthis line is duplicated.'}]


def test_newline_removal_modifier():
    modifier = newline_removal_modifier()  # Get the actual modification function
    modifier_single_newlines = newline_removal_modifier(max_consecutive=1)  # Get the actual modification function

    # Page with empty CONTENT
    empty_page = {CONTENT: ''}
    assert modifier(empty_page) == [{CONTENT: ''}]
    assert modifier_single_newlines(empty_page) == [{CONTENT: ''}]

    # Page with no newlines
    no_newline_page = {CONTENT: 'Hello, World'}
    assert modifier(no_newline_page) == [{CONTENT: 'Hello, World'}]
    assert modifier_single_newlines(no_newline_page) == [{CONTENT: 'Hello, World'}]

    # Page with 2 newlines
    two_newline_page = {CONTENT: 'Hello\n\nWorld'}
    assert modifier(two_newline_page) == [{CONTENT: 'Hello\n\nWorld'}]
    assert modifier_single_newlines(two_newline_page) == [{CONTENT: 'Hello\nWorld'}]

    # Page with more than 2 consecutive newlines
    many_newline_page = {CONTENT: 'Hello\n\n\n\n\nWorld'}
    assert modifier(many_newline_page) == [{CONTENT: 'Hello\n\nWorld'}]
    assert modifier_single_newlines(two_newline_page) == [{CONTENT: 'Hello\nWorld'}]

    # Page with non-consecutive newlines
    non_consecutive_page = {CONTENT: 'Hello\nWorld\nAnother\nLine'}
    assert modifier(non_consecutive_page) == [{CONTENT: 'Hello\nWorld\nAnother\nLine'}]
    assert modifier_single_newlines(non_consecutive_page) == [{CONTENT: 'Hello\nWorld\nAnother\nLine'}]

    # Page with non-consecutive and consecutive newlines
    mixed_page = {CONTENT: 'Hello\n\n\nWorld\n\n\nAnother\n\n\nLine'}
    assert modifier(mixed_page) == [{CONTENT: 'Hello\n\nWorld\n\nAnother\n\nLine'}]
    assert modifier_single_newlines(mixed_page) == [{CONTENT: 'Hello\nWorld\nAnother\nLine'}]


def test_split_lines_modifier():
    # Typical use case, when delimiter is \n
    page = {CONTENT: 'line1\nline2\n\nline3'}
    assert split_lines_modifier(page) == [{CONTENT: ['line1', 'line2', '', 'line3']}]

    # Changing the delimiter to \n\n
    page = {CONTENT: 'line1\nline2\n\nline3'}
    assert split_lines_modifier(page, delimiter = '\n\n') == [{CONTENT: ['line1\nline2', 'line3']}]

    # When CONTENT is already in the form of a list, do not make a modification
    page = {CONTENT: ['line1', 'line2', 'line3']}
    assert split_lines_modifier(page, delimiter = '\n\n') == [{CONTENT: ['line1', 'line2', 'line3']}]

    # When page is empty, simply remove
    page = {CONTENT: ''}
    assert split_lines_modifier(page) == []


def test_join_lines_modifier():
    # Typical use case, when delimiter is \n
    page = {CONTENT: ['line1', 'line2', '', 'line3']} 
    assert join_lines_modifier(page) == [{CONTENT: 'line1\nline2\n\nline3'}]

    # Changing the delimiter to \n\n
    page = {CONTENT: ['line1\nline2', 'line3']}
    assert join_lines_modifier(page, delimiter = '\n\n') == [{CONTENT: 'line1\nline2\n\nline3'}]

    # When CONTENT is already in the form of a list, do not make a modification
    page = {CONTENT: 'line1\nline2\n\nline3'}
    assert join_lines_modifier(page, delimiter = '\n\n') == [{CONTENT:  'line1\nline2\n\nline3'}]

    # When page is empty, simply remove
    page = {CONTENT: []}
    assert join_lines_modifier(page) == []
