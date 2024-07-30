'''
This script classifies a given text as either 'CC' or 'Wikipedia' using a FastText model from the RedPajama project.
More details about the model can be found in this GitHub issue: https://github.com/togethercomputer/RedPajama-Data/issues/24
Model download is done via `setup.py` script.
The download link is: https://drive.google.com/file/d/1DnsfpWWE0jFPCoYe6clwqb3Ub5Ac92s1/view?usp=share_link
'''
import os
from typing import Dict, List, Callable

import fasttext

from core.constants import CONTENT
from core.factory_utils import factory_function

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_SUBDIRECTORY = "baselines/mappers/enrichers/quality_prediction_enrichment_models"
RPJ_MODEL_FILENAME = 'model.bin'

def load_fasttext_model(model_filename):
    if os.path.exists(MODEL_SUBDIRECTORY):
        model_path = os.path.join(MODEL_SUBDIRECTORY, model_filename)
    else:
        model_path = os.path.join(PROJECT_ROOT, MODEL_SUBDIRECTORY, model_filename)

    assert os.path.exists(model_path), (
        f"Model {model_path} does not exist. "
        "Please download the model to this path before running a baselines pipeline involving fasttext filtering. "
        "See https://github.com/mlfoundations/dclm/blob/main/baselines/README.md#fasttext-filtering for more details."
    )

    return fasttext.load_model(model_path)

def classify_fasttext_hq_prob(model: fasttext.FastText._FastText, content: str) -> dict:
    '''
    This function classifies a given text as either 'CC' or 'Wikipedia' and returns the label along with its probability.

    Parameters:
    model (fasttext.FastText._FastText): The FastText model to use for the classification.
    content (str): The text to classify.

    Returns:
    dict: A value for 'hq_prob' - the probability to be a high-quality page.
    '''

    # Initialize an empty dictionary for the output
    output = {}

    # Clean the input text by joining all lines into a single string
    text = " ".join(content.strip().splitlines())

    # Make the prediction
    pred = model.predict(text)

    # Extract the predicted label and its probability
    (pred_label, pred_prob) = pred
    pred_label = pred_label[0]
    hq_prob = pred_prob[0]

    # If the predicted label is 'CC', adjust the probability of it being 'Wikipedia'
    if pred_label == "__label__cc":
        hq_prob = 1 - hq_prob

    # Return the output
    return hq_prob


@factory_function
def classify_fasttext_hq_prob_enricher(model_filename=RPJ_MODEL_FILENAME, key: str = "fasttext_hq_prob", overwrite: bool = False) -> Callable[
    [Dict], List[Dict]]:
    '''
    Enriches the given page with the text type (CC or Wikipedia).

    Parameters:
        page (dict): The page to enrich.
        model_filename (str): The name of the fasttext model file. Assumes it is placed in MODEL_SUBDIRECTORY.
        key (str): The key to store the text type under.
        overwrite (bool): Whether to overwrite the existing value of the key.

    Returns:
        A function that enriches the given page with the text type (HQ or CC).
    '''
    model = load_fasttext_model(model_filename)

    def enrich(page: Dict) -> List[Dict]:
        assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
        page[key] = classify_fasttext_hq_prob(model, page[CONTENT])
        return [page]

    return enrich
