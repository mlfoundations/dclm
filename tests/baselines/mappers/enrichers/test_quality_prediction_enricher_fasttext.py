import os

import pandas as pd

from baselines.core.constants import CONTENT
from baselines.mappers.enrichers.quality_prediction_enrichers_calc_fasttext import \
    classify_fasttext_hq_prob_enricher

current_directory = os.path.dirname(os.path.abspath(__file__))
test_files_path = os.path.join(current_directory, 'enricher_test_files')

def test_classify_fasttext_hq_prob_enricher():
    wiki_probs = pd.DataFrame(index=range(1, 6), columns=range(1, 6))

    for i in range(1, 6):  # assuming files are numbered from 1 to 5
        with open(os.path.join(test_files_path, f'wikipedia_paragraph{i}.html'), 'r') as file:
            wikipedia_content_page = {CONTENT: file.read()}

        result_wiki_page = classify_fasttext_hq_prob_enricher()(wikipedia_content_page)
        result_wiki = result_wiki_page[0]['fasttext_hq_prob']

        for j in range(1, 6):  # comparing each Wikipedia file to all CommonCrawl files
            with open(os.path.join(test_files_path, f'common_crawl_paragraph{j}.html'), 'r') as file:
                commoncrawl_content_page = {CONTENT: file.read()}

            result_cc_page = classify_fasttext_hq_prob_enricher()(commoncrawl_content_page)
            result_cc = result_cc_page[0]['fasttext_hq_prob']

            # store probability in DataFrame
            wiki_probs.at[i, j] = result_wiki - result_cc

            # Check if Wikipedia probability is higher
            assert result_wiki > result_cc, "Wikipedia probability is not higher than CommonCrawl probability"
