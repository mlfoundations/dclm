import pandas as pd

from baselines.mappers.enrichers.quality_prediction_enrichers_kenlm_model import *


def test_low_example():
    example_page = {CONTENT: "I am very perplexed"}
    example_page_enriched = ken_lm_perplexity_enricher()(example_page)
    ''' assert score around 341.3 '''
    assert abs(example_page_enriched[0]['kenlm_perplexity'] - 341.3) <= 0.1


def test_high_example():
    example_page = {CONTENT: "im hella trippin"}
    example_page_enriched = ken_lm_perplexity_enricher()(example_page)
    ''' assert score around 46793.5 '''
    assert abs(example_page_enriched[0]['kenlm_perplexity'] - 46793.5) <= 0.1


def test_get_perplexity():
    high_perplexity_sentences = [
        "Happily water you the pass can?",
        "To the store to go I need.",
        "The now time is what?",
        "Really good is weather the.",
        "Today very am I happy."
    ]

    low_perplexity_sentences = [
        "I am very happy today.",
        "The weather is really good.",
        "I am very perplexed",
        "What is the time now?",
        "I need to go to the store."
    ]

    high_perplexity_sentences_pages = [{CONTENT: sentence} for sentence in high_perplexity_sentences]
    low_perplexity_sentences_pages = [{CONTENT: sentence} for sentence in low_perplexity_sentences]

    low_perplexity_scores = [ken_lm_perplexity_enricher()(page)[0]['kenlm_perplexity'] for page in
                             low_perplexity_sentences_pages]
    high_perplexity_scores = [ken_lm_perplexity_enricher()(page)[0]['kenlm_perplexity'] for page in
                              high_perplexity_sentences_pages]
    assert min(high_perplexity_scores) > max(low_perplexity_scores), "Some high perplexity scores were not higher than low perplexity scores."

    # Ensure that each high_perplexity_scores[i] is higher than each low_perplexity_scores[j]
    for i in range(len(high_perplexity_sentences)):
        for j in range(len(low_perplexity_sentences)):
            assert high_perplexity_scores[i] > low_perplexity_scores[
                j], f"High perplexity sentence '{high_perplexity_sentences[i]}' ({high_perplexity_scores[i]}) was not higher than low perplexity sentence '{low_perplexity_sentences[j]}' ({low_perplexity_scores[j]})"

    # Calculate and output a dataframe with the perlexity gap between each of the pairs
    perplexity_gap_matrix = [[high - low for low in low_perplexity_scores] for high in high_perplexity_scores]
    df = pd.DataFrame(perplexity_gap_matrix,
                      columns=low_perplexity_sentences,
                      index=high_perplexity_sentences).T
    # Assert that all values in the dataframe are positive (i.e., perplexity gap is valid)
    assert (
                df.values > 0).all(), "Some high perplexity sentences had a lower score than expected when compared to low perplexity sentences."
