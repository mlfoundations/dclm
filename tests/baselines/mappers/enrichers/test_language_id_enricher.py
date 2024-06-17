from baselines.mappers.enrichers.language_id_enrichers import *

text1 = "War doesn't show who's right, just who's left."
text2 = "Ein, zwei, drei, vier"
text3 = "באתי, ראיתי, כבשתי"
text4 = "This is English.\nDies ist Deutsch, die offizielle Sprache in Deutschland.\nזה עברית"
empty_text = ""
spaces_text = "     "
symbols_text = "!@#$%^&*()_+"
mixed_line_text = "This is an English text that should be long. Dies ist Deutsch."
mixed_line_text_reversed = "Dies ist Deutsch. This is an English text that should be long."
mixed_line_text_de = "English. Dies ist Deutsch."
tokenizer = 'nltk'

fasttext_model = load_fasttext_model()

def test_single_line_lang_detect():
    res1 = detect_lang_paragraph_helper(text1, get_langdetect_lang_prob, tokenizer)
    assert list(res1.keys()) == ['en']
    assert [round(x['average_probability'], 2) for x in res1.values()] == [1.0]

    res2 = detect_lang_paragraph_helper(text2, get_langdetect_lang_prob, tokenizer)
    assert list(res2.keys()) == ['de']
    assert [round(x['average_probability'], 2) for x in res2.values()] == [1.0]

    res3 = detect_lang_paragraph_helper(text3, get_langdetect_lang_prob, tokenizer)
    assert list(res3.keys()) == ['he']
    assert [round(x['average_probability'], 2) for x in res3.values()] == [1]


def test_single_line_fast_text_enricher():
    res1 = detect_lang_paragraph_helper(text1, get_fasttext_lang_prob, tokenizer, fasttext_model)
    assert list(res1.keys()) == ['en']
    assert [round(x['average_probability'], 2) for x in res1.values()] == [0.99]

    res2 = detect_lang_paragraph_helper(text2, get_fasttext_lang_prob, tokenizer, fasttext_model)
    assert list(res2.keys()) == ['de']
    assert [round(x['average_probability'], 2) for x in res2.values()] == [0.95]

    res3 = detect_lang_paragraph_helper(text3, get_fasttext_lang_prob, tokenizer, fasttext_model)
    assert list(res3.keys()) == ['he']
    assert [round(x['average_probability'], 2) for x in res3.values()] == [1]


def test_multiple_lines():
    assert list(detect_lang_paragraph_helper(text4, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == ['en', 'de', 'he']
    assert list(detect_lang_paragraph_helper(text4, get_langdetect_lang_prob, tokenizer).keys()) == ['en', 'de', 'he']

def test_empty_string():
    assert list(detect_lang_paragraph_helper(empty_text, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == []
    assert list(detect_lang_paragraph_helper(empty_text, get_langdetect_lang_prob, tokenizer).keys()) == []


def test_spaces_string():
    assert list(detect_lang_paragraph_helper(spaces_text, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == []
    assert list(detect_lang_paragraph_helper(spaces_text, get_langdetect_lang_prob, tokenizer).keys()) == []


def test_symbols_string():
    assert list(detect_lang_paragraph_helper(symbols_text, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == []
    assert list(detect_lang_paragraph_helper(symbols_text, get_langdetect_lang_prob, tokenizer).keys()) == []


def test_mixed_line():
    assert list(detect_lang_paragraph_helper(mixed_line_text_de, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == ['en','de']
    assert list(detect_lang_paragraph_helper(mixed_line_text_de, get_langdetect_lang_prob, tokenizer).keys()) == ['en','de']
    
    assert list(detect_lang_paragraph_helper(mixed_line_text, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == ['en','de']
    assert list(detect_lang_paragraph_helper(mixed_line_text, get_langdetect_lang_prob, tokenizer).keys()) == ['en','de']

    assert list(detect_lang_paragraph_helper(mixed_line_text_reversed, get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == ['de','en']
    assert list(detect_lang_paragraph_helper(mixed_line_text_reversed, get_langdetect_lang_prob, tokenizer).keys()) == ['de', 'en']


def test_non_latin_scripts():
    assert list(detect_lang_paragraph_helper("你好，世界", get_fasttext_lang_prob, tokenizer, fasttext_model).keys()) == [
        'zh']  # fasttext detects the Chinese language as "zh
    assert list(detect_lang_paragraph_helper("你好，世界", get_langdetect_lang_prob, tokenizer).keys()) == [
        'zh-cn']  # langdetect detects the Chinese language as "zh-cn


def test_mixed_languages_in_lines():
    assert list(detect_lang_paragraph_helper("""This is English.\nDies ist Deutsch.""", get_fasttext_lang_prob,
                                             tokenizer, fasttext_model).keys()) == ['en', 'de']

    assert list(detect_lang_paragraph_helper("""This is English.\nDies ist Deutsch.""", get_langdetect_lang_prob,
                                             tokenizer).keys()) == ['en', 'de']


def test_language_whole_page():
    assert 'de' in list(detect_lang_whole_page_langdetect(text4).keys())
    assert 'de' in list(detect_lang_whole_page_fasttext(fasttext_model, text4).keys())


def test_language_whole_page_enricher():
     example_page = {CONTENT: text4}
     result = detect_lang_whole_page_enricher(LANGDETECT)(example_page)
     res_dict = result[0]['language_id_whole_page_langdetect']
     assert list(res_dict.keys()) == ['de']
     assert round(list(res_dict.values())[0], 1) == 1.0

     result = detect_lang_whole_page_enricher(FASTTEXT)(example_page)
     res_dict = result[0]['language_id_whole_page_fasttext']
     assert list(res_dict.keys()) == ['de']
     assert round(list(res_dict.values())[0], 1) == 0.9


def test_language_paragraph_enricher_fast_text():
    example_page = {CONTENT: text4}
    model_func_fast_text = detect_lang_paragraph_enricher(FASTTEXT, tokenizer)
    res_dict_fast_text = model_func_fast_text(example_page)
    res_dict = res_dict_fast_text[0]['language_id_paragraph_fasttext']
    assert list(res_dict.keys()) == ['en', 'de', 'he']
    assert [round(x['average_probability'], 2) for x in res_dict.values()] == [0.98, 0.99, 1.0]


def test_language_paragraph_enricher_lang_detect():
    example_page = {CONTENT: text4}
    res_dict_fast_text = detect_lang_paragraph_enricher(LANGDETECT, tokenizer)(example_page)
    res_dict = res_dict_fast_text[0]['language_id_paragraph_langdetect']
    assert list(res_dict.keys()) == ['en', 'de', 'he']
    assert [round(x['average_probability'], 2) for x in res_dict.values()] == [1.0, 1.0, 1.0]
