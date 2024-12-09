# In-depth Descriptions of Mappers, Filters, and Modifiers

## Table of Contents
1. [Introduction](#introduction)
2. [Key Concepts for Processing](#key-concepts-for-processing)
3. [Example YAML Configuration for Processing](#example-yaml-configuration-for-processing)
4. [Using Custom Mappers](#using-custom-mappers)
5. [Factory Functions](#factory-functions)
6. [Running the Processing Pipeline](#running-the-processing-pipeline)
7. [Setting Up a Ray Cluster](#setting-up-a-ray-cluster)
8. [Sample Workflows](#sample-workflows)

## Introduction
This document provides detailed descriptions of the key concepts involved in the data processing pipeline of the DCLM framework, including mappers, filters, and modifiers. It also includes instructions on setting up and running the processing pipeline using Ray clusters.

Before starting, ensure you have the necessary environment variables and configurations for AWS and Ray clusters as explained [here](../README.md).

## Key Concepts for Processing

### Mappers
Local operations applied to individual pages.
- **Example Operations**: Lowercasing text, removing HTML tags.
- **Code Location**: `baselines/mappers/`

### Filters
Operations that keep or remove pages based on certain criteria.
- **Example Operations**: Removing pages with low word count.
- **Code Location**: `baselines/filters/`

### Modifiers
Operations that modify the content within a page.
- **Example Operations**: Replacing specific words or phrases.
- **Code Location**: `baselines/modifiers/`

### Enrichers
Operations that add metadata to a page JSON.
- **Example Operations**: Language identification (LID) prediction.
- **Code Location**: `baselines/enrichers/`

### Global Functions
Operations that depend on all pages in the dataset.
- **Example Operations**: Deduplication, calculating global statistics like percentiles for perplexity scores.
- **Code Location**: `baselines/global_functions/`

### Aggregation Functions
While the majority of the processing is done independently per page, in some situations, it is useful to get some 
high-level view of results on a per-shard basis (i.e. a set of pages contained in one jsonl). For example, after performing language detection 
enrichment, you may want to examine the distribution of detected languages by computing a histogram of the classifications. 
Specifically, aggregators can compute summary stats based on the value of a chosen key (within a single shard).

For this purpose, we provide a few default [aggregation functions](aggregators.py) including percentiles and histogram 
(categorical or continuous). We also support a transformation function for the value associated with the enriched key in each page before it is passed 
to the aggregation function.  As in the mappers case, you can supply custom transform and aggregation functions as well.

To apply these functions, you add the `_aggregate` key to a step, and then name each aggregation function and provide a dict of arguments to pass to it.
For example, the [c4](baselines_configs/c4.yaml) pipeline contains this step:
```yaml
    - func: detect_lang_whole_page_enricher
      model: langdetect
      key_prefix: language_id_whole_page
      seed: 0
      _aggregate:
        language_id_whole_page_langdetect:
          type: histogram
          transform: threshold_transform
          threshold: 0.99
          default: "unknown"
```
This will have the following effect: 
1. It will run the language detection enrichment phase on each page, and store the result (a dict from language to confidence) as the value corresponding to the `language_id_whole_page_langdetect` key.
2. It will then run each such result through the `threshold_transform` transformation function with `threshold=0.99`, thus outputting the argmax language, given that it is above 99%, otherwise outputting `unknown`. The `_aggreagte` key maps to a dict, where the keys are the enriched keys in the shard to aggregate over, and the values are list of transformations and aggregation functions.
3. Finally, all detected languages will be pooled into a list and will be transferred to the histogram aggregation function, which will output a histogram of the languages detected, storing it in the shard's stats file.

## Example YAML Configuration for Processing

Below is an example of a YAML configuration file that defines a sequence of processing steps, that will be applied to inputs from the raw source cc_april_2019 (common crawl dump of April 2019).
Note - the source name is user-defined, and is used by [process_single_file](core/processor.py) to use the correct pipeline from a config file. A single config file can have different processing pipelines for different sources (e.g., common crawl, GitHub, Arxiv etc.).

```yaml
- source: cc_april_2019
  steps:
    - func: key_name_modifier  #  Changes the name of a key in a page dictionary
      old_key: content
      new_key: text
    - func: page_length_filter  # Filters the input JSON object based on the length of the CONTENT field.
      length_type: char
      max_length: 190000
    - func: word_length_modifier # Remove lines where the word with the largest length goes strictly over max_length. 
      max_length: 1000
      model: split
    - func: detect_lang_whole_page_enricher  # classify inputs' languages and store it under language_id_whole_page_{model}
      model: langdetect
      key_prefix: language_id_whole_page
    - func: language_filter  # removes lines where the classification probability of the content to be english is less than 0.99
      key: language_id_whole_page_langdetect
      keep_languages: [ en ]
      threshold: 0.99
```

For a full example of our reproduction of C4, see this [pipeline](baselines_configs/c4.yaml).

## Using Custom Mappers
By default, the pipeline configuration YAML can reference any mapper defined under [mappers](mappers), as detailed
[here](mappers/README.md). If you wish to use a custom mapper, you can specify it in the `func`
argument by providing a relative path from the working directory, separated with dots ('.'),
and including the mapper function name. For example, if you define a module `custom_mappers/my_filters.py` with a
mapper named `foo_filter`, you would set `func: custom_mappers.my_filters.foo_filter` in the YAML file.

## Factory Functions
Factory functions are used to create instances of mappers, filters, modifiers, and enrichers where some initialization needs to be done once and then reused for every application of the mapper. For example, when using regexes, they should be compiled once. Similarly, when using models, they should be loaded only once and not for every page. Factory functions are defined using the [factory_function](../core/factory_utils) decorator. In the function itself, you can load any necessary resources ahead of time, define a `filter_fn(page: Dict) -> List[Dict]` with a closure, and return the `filter_fn`. For an example, see `url_substring_filter` in [metadata_filters](mappers/filters/metadata_filters.py).

## Running the Processing Pipeline

To process raw data using the DCLM framework, follow these steps:

1.	Define a Set of Processing Steps: Create a pipeline config YAML file specifying the operations as shown in the example above.
2. Launch a Ray Cluster: Use an appropriate Ray cluster based on the size of your dataset and specific YAML configurations.
3. Run the Processing Script:

```bash
ray attach <your_cluster_config>
cd dcnlp
export PYTHONPATH=$(pwd)
python3 ray_processing/process.py \
  --source_ref_paths exp_data/datasets/raw_sources/CC_WET_april_2019.json \
  --readable_name c4 \
  --output_dir s3://dcnlp-west/cc_wet_2019_april_baselines/c4_v4/ \
  --config_path baselines/baselines_configs/c4.yaml \
  --source_name cc_april_2019
```
**Important Arguments**:

 - source_ref_paths: Path to reference JSON for a particular source. This json contains information about the source of the data, and where it is located. [Example](../exp_data/datasets/raw_sources/CC_WET_april_2019.json).
 - readable_name: Fills in the “name” field in the output JSON of the untokenized data. For example, [here](../exp_data/datasets/tokenized/c4_original.json) is the json file for a c4 reproduction, after it was tokenized.
 - output_dir: Path on S3 folder which will store the processed JSONL files (i.e. each input JSONL file that contains mlutiple pages from the raw source will be transformed to a processed JSONL file with a mirroring hierarchy, under this root dir).
 - config_path: Path to the YAML specifying the desired processing steps. For example, see the [config to reproduce C4](baselines_configs/c4.yaml).
 - source_name: Which source in the config_path YAML file to process. As mentioned above, this user defined key tells the processor which pipeline in the yaml config to use.

When processing a single shard (a JSONL file with multiple pages), the output result will be stored as a 
corresponding jsonl file with the same name and same relative path to the output dir as the input json to the source 
root, with a `_processed` suffix in the name.
Additionally, each processed shard will have corresponding `_stats.jsonl` file which will contain information on 
each step such as how many pages were processed in this step, how many were filtered out, how much time it took, etc. 
This file also allows to continue processing a shard if it was interrupted in the middle, from the last `commit` step 
(each commit will result in flushing the current state of the processed shard into storage).
When processing multiple shards, there will also be computed a global_stats file that will merge the stats of all shards.

## Setting Up a Ray Cluster

A Ray cluster is used for distributed processing of the data, using the pipeline defined above. Below are the steps to launch a Ray cluster on EC2 instances, run the desired job and tear down the cluster. **Make sure that you keep in mind the costs associated with the instances that are launched for your cluster, and that you tear down the cluster when you no longer need it!**. 
Additional instructions on how to deploy Ray on different setups can be found [here](../README.md#processing-the-data)

### Modify Cluster Config File

Below is a sample yaml file that defines a cluster in the `us-west-2` AWS region. Doing the edits outlined below allows for a cluster to be launched with this file.

```yaml
cluster_name: test-processing
max_workers: 2
upscaling_speed: 1.0
available_node_types:
    ray.head.default:
        resources: {}
        node_config:
            ImageId: ami-0c5cce1d70efb41f5
            InstanceType: i4i.4xlarge
            IamInstanceProfile:
                # Replace 000000000000 with your IAM account 12-digit ID
                Arn: arn:aws:iam::000000000000:instance-profile/ray-autoscaler-v1
    ray.worker.default:
        min_workers: 2
        max_workers: 2
        node_config:
            ImageId: ami-0c5cce1d70efb41f5
            InstanceType: i4i.4xlarge
            IamInstanceProfile:
                # Replace 000000000000 with your IAM account 12-digit ID
                Arn: arn:aws:iam::000000000000:instance-profile/ray-autoscaler-v1

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    cache_stopped_nodes: False

setup_commands:
    - sudo mkfs -t xfs /dev/nvme1n1
    - sudo mount /dev/nvme1n1 /tmp
    - sudo chown -R $USER /tmp
    - sudo chmod -R 777 /tmp
    - wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O miniconda.sh
    - bash ~/miniconda.sh -f -b -p /tmp/miniconda3/
    - echo 'export PATH="/tmp/miniconda3/bin/:$PATH"' >> ~/.bashrc
    # Include your AWS CREDS here
    - echo 'export AWS_ACCESS_KEY_ID=' >> ~/.bashrc
    - echo 'export AWS_SECRET_ACCESS_KEY=' >> ~/.bashrc
    - pip install --upgrade pip setuptools wheel
    - pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
    - pip install boto3==1.26.90
    - pip install s3fs==2022.11.0
    - pip install psutil
    - pip install pysimdjson
    - pip install pyarrow
    - git clone https://github.com/mlfoundations/dclm.git
    - pip install -r dclm/requirements.txt
    - cd dclm && python3 setup.py install
```

### Launch the Cluster

```bash
ray up <your_cluster_config>
```

### Attach to the Head Node and Run Processing

```bash
ray attach <your_cluster_config>
cd dcnlp
export PYTHONPATH=$(pwd)
screen -S processing
python3 ray_processing/process.py \
  --source_ref_paths exp_data/datasets/raw_sources/CC_WET_april_2019.json \
  --readable_name c4_v4 \
  --output_dir s3://dcnlp-west/cc_wet_2019_april_baselines/c4_v4/ \
  --config_path baselines/baselines_configs/c4.yaml \
  --source_name cc_april_2019
```

### Monitor Progress and Tear Down
 - Check progress via global_stats.jsonl in the output directory.
 - Tear down the cluster after processing:

```bash
ray down <your_cluster_config>
```

## Sample Workflows

### Fasttext Filtering

Running model-based filtering with our processor typically involves using two mappers. First is an *enricher* that handles the model inference and adds quality scores to each page. Second is a *filter* that thresholds these scores and removes documents. 

See [this](baselines_configs/fasttext_filter.yaml) for an example that corresponds to the specific OH2.5 + ELI5 classifier that we use for DCLM-Baseline. Notably, the steps involved are. 

```
  steps:
    - func: classify_fasttext_hq_prob_enricher
      model_filename: fasttext_oh_eli5.bin  # Change this to the name of your model file
      key: fasttext_oh_eli5_vs_rw_v2_prob   # Change this to the name of the desired key
    - func: quality_filter
      key: fasttext_oh_eli5_vs_rw_v2_prob   # Make sure this matches with the key from the enricher
      threshold: 0.018112                   # Chnage this to your chosen threshold.
```

Important Notes:
- In many scenarios, such as when you wish to tune the threshold, it may make sense to save the outputs of the first enriching step as an intermediate dataset to avoid having to repeatedly run inference on the same pages. This can be done by placing these two mappers in separate processing pipelines (i.e., yaml files) instead of the same one.
- It is assumed that the fasttext model has been downloaded and available in `baselines/mappers/enrichers/quality_prediction_enrichment_models` on all nodes _prior_ to invoking the baselines processor `process.py`. If setting up a ray cluster, a natural strategy would be to place model downloads within the `setup_commands` of your ray cluster config. As an example, our [`setup.py`](../setup.py#L107) contains code for downloading  the OH2.5 + ELI5 classifier from HuggingFace so we simply add `python setup.py install` as one of our `setup_commands` steps.
- You may need to increase `--ray_num_cpus` to be avoid running into memory issues (since this mapper involves loading/running a fasttext model which can be multiple GBs). We use `--ray_num_cpus 2` with EC2 `i4i.4xlarge` ndoes.
- Do not use `--ray_use_working_dir` when running this step or your ray tasks may have trouble accessing the model binary.
- For filtering, we currently share the threshold used in the DCLM-Baseline. A (global) function for aggregating scores and dynamically computing percentile-based thresholds will be coming soon.

__Training your own fasttext classifiers:__ A simple script for training fasttext classifiers can be found [here](train_fasttext_classifier.py). This is a basic wrapper over the [`fasttext`](https://fasttext.cc/docs/en/supervised-tutorial.html) package's `train_supervised` function. For example, the following command

  ```
  python train_fasttext_classifier.py --input your_train_data.txt --name your_model_name --wordNgrams 2
  ```

  will produce a model called `your_model_name.bin` and saved in the default directory `/mappers/enrichers/quality_prediction_enrichment_models/`. Here, we add `--wordNgrams 2` to indicate that we wish to use both word-level unigrams and bigrams as features. Other fasttext hyperparameters are also supported by our script but we did not deviate from their default values for our OH2.5 + ELI5 classifier.

  As described in the [fasttext documentation](https://fasttext.cc/docs/en/supervised-tutorial.html#getting-and-preparing-the-data), the file `your_train_data.txt` must be a text file where each line corresponds to one training example and which starts with the the label denoted by prefix `__label__`. In our codebase, we assume the label space must be `__label__hq` for "high-quality"  and `__label__cc` for "low-quality". Thus, the training data would follow the following format.

  ```
  __label__hq this is a good sentence
  __label__cc this is a bad sentence
  ```
