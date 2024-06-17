import pytest
import os
from baselines.mappers.filters import *
import random


def test_random_sampling_filter():
    page = {CONTENT: "content"}

    for seed in [1,2,3]:
        random.seed(seed)
        pages = [random_sampling_filter(page, 0.1) for i in range(10000)]
        assert 900 <= sum(len(p) for p in pages) <= 1100

    with pytest.raises(AssertionError):
        assert random_sampling_filter(page, 1.1)
    with pytest.raises(AssertionError):
        assert random_sampling_filter(page, -0.1)


def test_url_substring_filter():
    # Test basic presence/absence of the banned word
    page = {URL: 'https://www.badword.com'}
    assert url_substring_filter(['badword', 'reallybadword'])(page) == []
    assert url_substring_filter(['reallybadword', 'extremelybadword'])(page) == [page]

    # Test robustness to urls that split up a bad word to bypass detectors
    page = {URL: 'https://www.ba-dwo.rd.com'}
    assert url_substring_filter('badword')(page) == [page]
    assert url_substring_filter(['badword'])(page) == [page]
    assert url_substring_filter('badword', ignore_chars=['-', '.'])(page) == []
    assert url_substring_filter(['badword'], ignore_chars=['-', '.'])(page) == []

    # Test when num_banned_substrs > 1
    page = {URL: 'https://www.badword123-badword2.com'}
    assert url_substring_filter(['badword1', 'badword2', 'badword3'], num_banned_substrs=3)(page) == [page]
    assert url_substring_filter(['badword1', 'badword2', 'badword3'], num_banned_substrs=3, match_substrings=False)(page) == [page]
    assert url_substring_filter(['badword1', 'badword2', 'badword3'], num_banned_substrs=2)(page) == []
    assert url_substring_filter(['badword1', 'badword2', 'badword3'], num_banned_substrs=2, match_substrings=False)(page) == [page]
    assert url_substring_filter(['badword123', 'badword2', 'badword3'], num_banned_substrs=2, match_substrings=False)(page) == []
    # In current implementation, banlist items must not be overlapping
    assert url_substring_filter(['badword1', 'badword2', 'badword3', 'word123'], num_banned_substrs=3)(page) == [page]
    
    # Test case sensitivity
    page = {URL: 'https://www.badword.com/BadWord2'}
    assert url_substring_filter('BadWord2')(page) == []
    assert url_substring_filter('badword2')(page) == []
    assert url_substring_filter('BadWord2', case_sensitive=True)(page) == []
    assert url_substring_filter('badword2', case_sensitive=True)(page) == [page]
    assert url_substring_filter(['badword', 'Bad'], num_banned_substrs=2, match_substrings=True, case_sensitive=True)(page) == []
    assert url_substring_filter(['badword', 'bad'], num_banned_substrs=2, match_substrings=True, case_sensitive=True)(page) == [page]
    assert url_substring_filter(['badword', 'Bad'], num_banned_substrs=2, match_substrings=False, case_sensitive=True)(page) == [page]
    assert url_substring_filter(['badword', 'BadWord2'], num_banned_substrs=2, match_substrings=False, case_sensitive=True)(page) == []
    assert url_substring_filter(['badword', 'badword2'], num_banned_substrs=2, match_substrings=False, case_sensitive=True)(page) == [page]
    
    #Test using the url substring filter
    fname = 'tests/baselines/mappers/filters/banlists/refinedweb_banned_domains_and_urls.txt' if os.getcwd().endswith("/dcnlp") else 'banlists/refinedweb_banned_domains_and_urls.txt'
    pickle_fname = 'tests/baselines/mappers/filters/banlists/refinedweb_banned_domains_and_urls_regex.pkl' if os.getcwd().endswith("/dcnlp") else 'banlists/refinedweb_banned_domains_and_urls_regex.pkl'
    domains_fname = 'tests/baselines/mappers/filters/banlists/refinedweb_banned_domains.txt' if os.getcwd().endswith("/dcnlp") else 'banlists/refinedweb_banned_domains_and_urls.txt'
    
    refinedweb_url_filter = url_substring_filter(banlist_from_fname=fname)
    refinedweb_url_filter_pickle = url_substring_filter(banlist_from_fname=pickle_fname)
    refinedweb_domain_filter = url_substring_filter(banlist_from_fname=domains_fname, exact_domain_match=True)

    # High-quality sources that should be kept in
    page = {URL: 'https://en.wikipedia.org/wiki/Large_language_model'}
    assert refinedweb_url_filter(page) == [page]
    assert refinedweb_url_filter_pickle(page) == [page]
    assert refinedweb_domain_filter(page) == [page]

    page = {URL: 'https://arxiv.org/abs/2304.14108'}
    assert refinedweb_url_filter(page) == [page]
    assert refinedweb_url_filter_pickle(page) == [page]
    assert refinedweb_domain_filter(page) == [page]

    page = {URL: 'https://github.com/mlfoundations/datacomp'}
    assert refinedweb_url_filter(page) == [page]
    assert refinedweb_url_filter_pickle(page) == [page]
    assert refinedweb_domain_filter(page) == [page]

    # le.com is in the refinedweb banlist of domains
    page = {URL: 'https://www.google.com/search?q=datacomp'}
    assert refinedweb_url_filter(page) == []
    assert refinedweb_url_filter_pickle(page) == []
    assert refinedweb_domain_filter(page) == [page]

    # Page that should be removed
    page = {URL: 'https://xxx.com/adult-video'}
    assert refinedweb_url_filter(page) == []
    assert refinedweb_url_filter_pickle(page) == []
    assert refinedweb_domain_filter(page) == []

    # Need to pass in 'www.' to ignore_chars 
    page = {URL: 'https://www.xxx.com'}
    assert refinedweb_url_filter(page) == []
    assert refinedweb_url_filter_pickle(page) == []
    assert refinedweb_domain_filter(page) == [page]
    assert url_substring_filter(banlist_from_fname=domains_fname, ignore_chars=['www.'], exact_domain_match=True)(page) == []
    
    # Test with empty page
    page = {}
    with pytest.raises(KeyError):
        url_substring_filter(['badword'])(page)


def test_language_filter():
    page_above_threshold = {
        CONTENT: "This is an example content in English.",
        "language_id_whole_page_langdetect": {"en": 0.81}
    }

    page_below_threshold = {
        CONTENT: "This is another example content in English.",
        "language_id_whole_page_langdetect": {"en": 0.79}
    }

    # Keeping English pages with probability above 0.8
    result = language_filter(page_above_threshold, ['en'], threshold=0.8)
    assert len(result) == 1, "Page with probability above threshold was not kept."

    # Discarding English pages with probability below 0.8
    result = language_filter(page_below_threshold, ['en'], threshold=0.8)
    assert len(result) == 0, "Page with probability below threshold was kept."

    # Check for missing key
    page_without_key = {CONTENT: "This is an example content."}
    try:
        result = language_filter(page_without_key, ['en'])
        assert False, "Page without key did not raise AssertionError."
    except AssertionError:
        pass


def test_language_filter_v2():
    # Test with language not in the keep_languages list
    page = {'language_id_whole_page_langdetect': {'en': 0.9}}
    assert language_filter(page, ['es', 'fr', 'de']) == []

    # Test with language in the keep_languages list but below threshold
    page = {'language_id_whole_page_langdetect': {'en': 0.5}}
    assert language_filter(page, ['es', 'fr', 'en', 'de'], threshold=0.8) == []

    # Test with language in the keep_languages list and above threshold
    page = {'language_id_whole_page_langdetect': {'en': 0.9}}
    assert language_filter(page, ['es', 'fr', 'en', 'de'], threshold=0.8) == [page]

    # Test with empty keep_languages list
    page = {'language_id_whole_page_langdetect': {'en': 0.9}}
    assert language_filter(page, []) == []

    # Test with no 'language_id_whole_page_langdetect' in page
    page = {}
    try:
        language_filter(page, ['es', 'fr', 'en', 'de'])
        assert False, "Page without key did not raise AssertionError."
    except AssertionError:
        pass

    # Test with keep_languages not as list
    page = {'language_id_whole_page_langdetect': {'en': 0.9}}
    try:
        language_filter(page, 'es')
        assert False, "keep_languages not as list did not raise TypeError."
    except TypeError:
        pass

    # Test with empty page
    page = {}
    try:
        language_filter(page, ['en'])
        assert False, "Empty page did not raise AssertionError."
    except AssertionError:
        pass

    # Test with language in keep_languages list but case mismatch
    page = {'language_id_whole_page_langdetect': {'EN': 0.9}}
    assert language_filter(page, ['en']) == []


def test_quality_filter():
    page = {CONTENT: "This is an example content.", "fasttext_hq_prob": 0.75, "kenlm_perplexity": 176.1}

    result = quality_filter(page, "fasttext_hq_prob", 0.74)
    assert len(result) == 1, "Page was not kept when threshold was equal to real quality score using 'fasttext'."

    result = quality_filter(page, "fasttext_hq_prob", 0.75)
    assert len(result) == 1, "Page was not kept when threshold was equal to real quality score using 'fasttext'."

    result = quality_filter(page, "fasttext_hq_prob", 0.9)
    assert len(result) == 0, "Page was kept when threshold was above real quality score using 'fasttext'."

    result = quality_filter(page, "kenlm_perplexity", 176.0, lower_better=True)
    assert len(result) == 0, "Page was kept when threshold was below real quality score using 'kenlm'."

    result = quality_filter(page, "kenlm_perplexity", 176.1, lower_better=True)
    assert len(result) == 1, "Page was not kept when threshold was equal to real quality score using 'kenlm'."

    result = quality_filter(page, "kenlm_perplexity", 177.0, lower_better=True)
    assert len(result) == 1, "Page was not kept when threshold was above real quality score using 'kenlm'."

    # Using unsupported key
    result = quality_filter(page, "unsupported_key", 0.8, key_must_exist=False)
    assert len(result) == 0, "Page did not have key but key_must_exist=False so it should simply be filtered out"

    try:
        result = quality_filter(page, "unsupported_key", 0.8)
        assert False, "Unsupported key did not raise ValueError."
    except AssertionError:
        pass


if __name__ == "__main__":
    pytest.main()
