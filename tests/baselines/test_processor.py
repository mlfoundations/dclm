import os
import shutil

from baselines.core.processor import process_single_file

import json
import pytest

TMP_OUTPUT_DIR = './tmpoutputs'
DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@pytest.fixture
def file_cleanup():
    os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

    yield

    # The cleanup code below will run regardless of the test outcome
    if os.path.exists(TMP_OUTPUT_DIR):
        shutil.rmtree(TMP_OUTPUT_DIR)


@pytest.mark.parametrize("workers", [1, 2],
                         ids=["Sequential processing", "Parallel processing"])
def test_main_with_args(file_cleanup, workers):
    assert "tests" not in os.getcwd(), 'Please run this test from the root directory of the project'
    yaml = os.path.join(DATA_BASE_PATH, 'example_config.yaml')
    raw_data_dirpath = os.path.join(DATA_BASE_PATH, 'example_data/root_data_dir')  # this will not be mirrored in the output dir hierarchy
    jsonl = 'subdir1/shard1.jsonl'  # but this will

    # Run the main function with the parsed arguments
    output_path, stats_output_path = process_single_file(
        config_path=yaml,
        raw_data_dirpath=raw_data_dirpath,
        jsonl_relpath=jsonl,
        source_name='mock_data',
        output_dir=TMP_OUTPUT_DIR,
        overwrite=True,
        workers=workers)

    # Verify the hierarchy of the output paths mirrors that of the input, but under the output dir
    assert output_path == os.path.join(TMP_OUTPUT_DIR, 'example_config', 'subdir1', 'shard1_processed.jsonl')
    assert stats_output_path == os.path.join(TMP_OUTPUT_DIR, 'example_config', 'stats', 'subdir1', 'shard1_stats.jsonl')

    # Read the expected results from a file
    with open(os.path.join(DATA_BASE_PATH, 'expected_output.jsonl'), 'r') as f:
        expected_results = [json.loads(line) for line in f]

    # Read the actual results from the output file
    with open(output_path, 'r') as f:
        actual_results = [json.loads(line) for line in f]

    # Compare the actual results to the expected results. In the case of the multiprocessing test, the order of the
    # lines may vary, and we need to account for that
    actual_results = sorted(actual_results, key=lambda x: x['url'])
    expected_results = sorted(expected_results, key=lambda x: x['url'])

    for ac in actual_results:
        ac['language_id_whole_page_langdetect'] = {k: round(v, 6) for k, v in ac['language_id_whole_page_langdetect'].items()}
    for ec in expected_results:
        ec['language_id_whole_page_langdetect'] = {k: round(v, 6) for k, v in ec['language_id_whole_page_langdetect'].items()}

    assert actual_results == expected_results

    # now test the stats file (ignoring time)
    # Read the expected results from a file
    def test_key(k):
        return "secs" not in k and "_time" not in k and "_path" not in k
    def clear_dict_vals(dict_val):
        if isinstance(dict_val, dict):
            return {k: v for k, v in dict_val.items() if test_key(k)}
        return dict_val

    with open(os.path.join(DATA_BASE_PATH, 'expected_stats.jsonl'), 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
        expected_keys = [set(line.keys()) for line in lines]
        expected_results = [{k: clear_dict_vals(v) for k, v in line.items() if test_key(k)} for line in lines]
        for line in expected_results:
            if 'workers' in line:
                line['workers'] = workers

    # Read the actual results from the output file
    with open(stats_output_path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
        actual_keys = [set(line.keys()) for line in lines]
        actual_results = [{k: clear_dict_vals(v) for k, v in line.items() if test_key(k)} for line in lines]

    # Compare the actual keys to the expected results.
    assert actual_keys == expected_keys

    # Compare the actual results to the expected results.
    assert actual_results == expected_results

# TODO - add a test for the commit function and continue from previous run
# TODO - need t make sure that factory functions are loaded only once per worker.


if __name__ == "__main__":
    pytest.main()
