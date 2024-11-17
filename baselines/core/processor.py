import logging
import multiprocessing
import os
import os.path
import time
from datetime import datetime
from typing import Any, Dict, Tuple

from yaml import safe_load

from baselines.core.factories import get_mapper, get_aggregator, get_transform
from baselines.core.file_utils import read_jsonl, write_jsonl, makedirs_if_missing, delete_file, is_exists
from baselines.core.constants import PROCESS_SETUP_KEY_NAME, PROCESS_END_KEY_NAME, COMMIT_KEY_NAME, GLOBAL_FUNCTIONS

logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)  # For example, set level to DEBUG

# Create a StreamHandler for stdout
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)  # Optionally set a different level for stdout

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stdout_handler.setFormatter(formatter)

# Add the stdout handler to the logger
logger.addHandler(stdout_handler)

COMMIT_COMMAND = 'commit'

# Indices in the mapper counters (what happened to each processed page)
ERRORS_INDEX = -1
REMOVED_INDEX = 0
KEPT_INDEX = 1
SPLIT_INDEX = 2


def commit(pages, stats, output_path, stats_output_path):
    logger.info(f"committing pages to {output_path} (and stats to {stats_output_path})")
    t = time.time()
    write_jsonl(pages, output_path)

    stats.append({'name': COMMIT_KEY_NAME, 'secs': time.time() - t})
    write_jsonl(stats, stats_output_path, 'a')

    stats.clear()
    logger.info(f"commit took {time.time() - t} seconds")


def _get_output_paths(base_output_path, jsonl_relpath):
    # TODO - need to allow overwrite?
    file_ext = os.path.splitext(jsonl_relpath)[-1][1:]
    file_ext = f'jsonl.{file_ext}' if file_ext in ['zstd', 'zst', 'gz'] else file_ext
    jsonl_relpath = jsonl_relpath.replace("_processed.jsonl", ".jsonl")    # To allow for overwrite if continuning from intermediate
    shard_name = jsonl_relpath.split(".jsonl")[0]
    out_path = os.path.join(base_output_path, 'processed_data', shard_name + f'_processed.{file_ext}')
    stats_out_path = os.path.join(base_output_path, 'stats', shard_name + '_stats.jsonl')

    makedirs_if_missing(os.path.dirname(out_path))
    makedirs_if_missing(os.path.dirname(stats_out_path))
    return out_path, stats_out_path

def _is_step_stats(line):
    """
    True iff this is a step stats line (and not a general info one)
    """
    return line['name'] not in {PROCESS_SETUP_KEY_NAME, PROCESS_END_KEY_NAME, COMMIT_KEY_NAME}


def process_single_file(config_data: Dict[str, Any], raw_data_dirpath: str, jsonl_relpath: str, source_name: str, 
                        base_output_path: str, workers: int = 1, overwrite: bool = False) -> Tuple[str, str]:
    """
    :param config_data: A processed config (from yaml) that specifies the steps to be taken
    :param raw_data_dirpath: The path to the top data directory in the data hierarchy (from which to mirror
                                the output path)
    :param jsonl_relpath: path to the input jsonl file with the text to be processed, relative to raw_data_dirpath
    :param source_name: The name of the source of the jsonl file
    :param base_output_path: path to the output directory where to save the resulting jsonl file with the
                        processed text as well as the stats
    :param workers: number of workers to use for multiprocessing. if 1, will use a single process
    :param overwrite: if True, will overwrite the output file if it already exists, otherwise will continue from
                        where previous runs stopped
    :return: The output path where the processed jsonl file was saved and the stats output path
    """
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    t0 = time.time()

    # Assumption #1 - we do not write the output after every step, but only at the end/on a commit step
    assert source_name in config_data, f"Source {source_name} not found in YAML file."
    config_data = config_data[source_name]

    # Assumption #2 - we keep the entire input lines in memory
    t1 = time.time()
    input_path = os.path.join(raw_data_dirpath, jsonl_relpath)
    pages = list(read_jsonl(input_path))
    jsonl_load_secs = time.time() - t1

    # Assumption #3 - for each input shard, there is a specific stats file that accompanies it
    output_path, stats_output_path = _get_output_paths(base_output_path, jsonl_relpath)

    # If a jsonl is empty (e.g., due to another chunk), the page will be skipped  
    num_pages_in = len(pages)
    if num_pages_in == 0:
        logger.info(f"Input data file at {input_path} is empty.")
        return output_path, stats_output_path, 0, 0

    # load stats file
    t2 = time.time()
    old_stats, stats = [], []  # we will use this for fault tolerance - if the stats file exists, we will
    # skip the steps that were already done

    if is_exists(stats_output_path):
        if overwrite:
            logger.info(f"Stats file {stats_output_path} already exists, but overwrite is set to true, so deleting.")
            delete_file(stats_output_path)
        else:
            logger.info(f"Stats file {stats_output_path} already exists, loading.")
            old_stats = list(read_jsonl(stats_output_path))
            # Note - we could simply count '\n' to know how many lines to skip and not read the whole thing,
            # but this is more robust and less error-prone
    stats_load_secs = time.time() - t2
    first_run = len(old_stats) == 0  # i.e. not a continuation
    graceful_continuation = len(old_stats) >= 4 and old_stats[-2]['name'] == PROCESS_END_KEY_NAME and \
                            old_stats[-1]['name'] == COMMIT_KEY_NAME  # i.e. the last run did at least 1 step +
    if not first_run:
        old_stats = [line for line in old_stats if _is_step_stats(line)]  # remove the setup, commit and end messages
    # (start message, end message and commit message) and the execution ended gracefully
    stats.append({'name': PROCESS_SETUP_KEY_NAME,
                  'jsonl_load_secs': jsonl_load_secs,
                  'stats_load_secs': stats_load_secs,
                  'jsonl_path': jsonl_relpath,
                  'source_name': source_name,
                  'output_path': output_path,
                  'stats_output_path': stats_output_path,
                  'workers': workers,
                  'first_run': first_run,
                  'graceful_continuation': graceful_continuation,
                  'execution_start_time': start_time})

    i = 0
    commit_steps = executed_steps = skipped_steps = 0
    early_exit = False  # TODO - will be used when we break out of the loop early, for example for dedup or GPU use
    updated = False  # a flag to signify we actually perform any step at all that can be commited
    for step in config_data['steps']:
        if step == COMMIT_COMMAND:
            if not updated:
                logger.info("No steps were executed, skipping commit.")
                continue
            commit_steps += 1
            commit(pages, stats, output_path, stats_output_path)
            updated = False
            continue

        # skip step if was done already
        if i < len(old_stats):
            assert old_stats[i]['name'] == step[
                'func'], f"Step {step} does not match step {old_stats[i]['name']} in stats file."
            logger.info(f"Skipping step {step} since it was already done.")
            i += 1  # does not count commit steps. and old_stats also doesn't hold their messages
            # TODO - Unresolved issue - if we commit and exit to use GPU, but, the metadata already exists and we don't need to overwrite, how to we skip this?
            skipped_steps += 1
            continue
        elif step['func'] in GLOBAL_FUNCTIONS:
            # Assumption: GLOBAL functions will do their own logging to the respective stats files, thus can always
            # just exit if we reach this elif condition, since completed GLOBAL fucntions will only reach previous if block. 
            early_exit = True
            break

        executed_steps += 1
        n_pages_before = len(pages)
        counters = {ERRORS_INDEX: 0, REMOVED_INDEX: 0, KEPT_INDEX: 0, SPLIT_INDEX: 0}
        new_pages = []
        execution_times = []
        step_stats = {}

        # Apply the function over all lines in the input JSONL iteratively
        # TODO - allow batching in a single mapper
        t4 = time.time()
        if workers > 1:
            apply_partial_func_parallel(counters, execution_times, new_pages, pages, step, workers)
        else:
            # Create the partial function for each item in the YAML
            # Assumption - if the function has some intiialization to do, it was done as part of a cluster creation and the overhead will be amortized across pages
            partial_func = get_mapper(**step, _profile=True, _safe=True)
            apply_partial_func_sequential(counters, execution_times, new_pages, pages, partial_func)
            del partial_func
        if counters[ERRORS_INDEX] == len(pages):
            raise RuntimeError(f"Step {step} failed on all pages.")
        t5 = time.time()

        n_pages_after = len(new_pages)
        step_stats['name'] = step['func']
        step_stats['pages_in'] = len(pages)
        step_stats['pages_out'] = len(new_pages)
        step_stats['secs'] = sum(execution_times)
        step_stats['secs/page'] = step_stats['secs'] / len(pages)
        step_stats['errors'] = counters[ERRORS_INDEX]
        step_stats['removed'] = counters[REMOVED_INDEX]
        step_stats['kept'] = counters[KEPT_INDEX]
        step_stats['split'] = counters[SPLIT_INDEX]
        step_stats['total_secs'] = t5 - t4
        step_stats['workers'] = workers

        logger.info(f"Step {step['func']} removed {counters[REMOVED_INDEX]} pages, split {counters[SPLIT_INDEX]} "
                    f"pages, and kept {counters[KEPT_INDEX]} pages. {counters[ERRORS_INDEX]} errors were detected. "
                    f"In total, {n_pages_before} pages before and {n_pages_after} pages after.")
        pages = new_pages

        if len(pages) == 0:
            logger.info("No pages left, stopping.")
            updated = True
            break

        if '_aggregate' in step:
            for agg_key, agg_def in step['_aggregate'].items():
                t_agg = time.time()
                if isinstance(agg_def, str):
                    # if it's a string, assume it's the aggregator type and that it needs no transform
                    agg_def = {'type': agg_def}
                agg_type = agg_def['type']
                agg = get_aggregator(agg_type)
                vals = [p[agg_key] for p in pages]
                if 'transform' in agg_def:
                    transform_name = agg_def['transform']
                    transform_args = {k: v for k, v in agg_def.items() if k != 'transform' and k != 'type'}
                    transform = get_transform(transform_name, **transform_args)
                    vals = [transform(v) for v in vals]
                agg_res = agg(vals)
                logger.info(f'Aggregation result for {agg_key} is {agg_res}')
                agg_res['_secs'] = time.time() - t_agg  # add how much time it took to run this aggregation
                step_stats[agg_key] = agg_res
        stats.append(step_stats)

        updated = True

    stats.append({'name': PROCESS_END_KEY_NAME,
                  'total_secs': time.time() - t0,
                  'executed_steps': executed_steps,
                  'skipped_steps': skipped_steps,
                  'commit_steps': commit_steps,
                  'finished_processing': not early_exit,
                  'execution_end_time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})

    logger.info('Finished processing all steps, committing.')
    if updated:
        commit(pages, stats, output_path, stats_output_path)

    return output_path, stats_output_path, num_pages_in, len(pages)


def _parse_func_results(results_gen, counters, execution_times, new_pages):
    for result, profiling_info in results_gen:
        execution_times.append(profiling_info.execution_time)
        if isinstance(result, list):
            counters[min(len(result), 2)] += 1  # 0 is removed, 1 is kept and 2 is split
            new_pages.extend(result)
        else:
            counters[ERRORS_INDEX] += 1  # error
            logger.error(result)


def apply_partial_func_sequential(counters, execution_times, new_pages, pages, partial_func):
    _parse_func_results(map(partial_func, pages), counters, execution_times, new_pages)


def _worker(worker_num, step, pages, input_queue, output_list):
    logger.info(f"Worker {worker_num} has started processing jobs.")
    # Create the partial function for each item in the YAML
    # Assumption - if the function has some intiialization to do, it was done as part of a cluster creation and the overhead will be amortized across pages
    partial_func = get_mapper(**step, _profile=True, _safe=True)
    while True:
        try:
            # Get a job from the input queue.
            page_ind = input_queue.get()

            if page_ind is None:
                # If the job is None, this signals the worker to terminate.
                break

            result = partial_func(pages[page_ind])

            # Put the result in the output queue.
            output_list.append(result)
        except Exception as e:
            logger.exception(f"Error occurred in MP apply of func: {e}")

    logger.info(f"Worker {worker_num} has finished processing jobs.")


def apply_partial_func_parallel(counters, execution_times, new_pages, pages, step, n_workers):
    # Create the input queue.
    input_queue = multiprocessing.Queue()

    # Create the shared list.
    manager = multiprocessing.Manager()
    shared_list = manager.list()

    # Create the workers.
    workers = [multiprocessing.Process(target=_worker, args=(i, step, pages, input_queue, shared_list)) for i in
               range(n_workers)]

    # Start the workers.
    for w in workers:
        w.start()

    # Add jobs to the input queue.
    for i in range(len(pages)):
        input_queue.put(i)

    # Add a None job to the end of the queue for each worker, to signal them to terminate.
    for _ in range(n_workers):
        input_queue.put(None)

    # Wait for all workers to finish.
    for w in workers:
        w.join()

    _parse_func_results(shared_list, counters, execution_times, new_pages)
