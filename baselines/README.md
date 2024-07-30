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

## Example YAML Configuration for Processing

Below is an example of a YAML configuration file that defines a sequence of processing steps:

```yaml
pipeline:
  - name: lowercase
    type: mapper
  - name: remove_html
    type: mapper
  - name: filter_length
    type: filter
    params:
      min_length: 100
  - name: enrich_language_id
    type: enricher
```

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
  --readable_name c4_v4 \
  --output_dir s3://dcnlp-west/cc_wet_2019_april_baselines/c4_v4/ \
  --config_path baselines/baselines_configs/c4.yaml \
  --source_name cc_april_2019
```
**Important Arguments**:

 - source_ref_paths: Path to reference JSON for a particular source.
 - readable_name: Fills in the “name” field in the output JSON.
 - output_dir: Path on S3 folder which will store the processed JSONLs.
 - config_path: Path to the YAML specifying the desired processing steps.
 - source_name: Which source in the config_path YAML file to process.

## Setting Up a Ray Cluster

### Modify Cluster Config File

```yaml
cluster_name: my-cluster
min_workers: 1
max_workers: 10
upscaling_speed: 1.0
docker:
  image: "rayproject/ray:latest"
  container_name: "ray_container"
  pull_before_run: True
initial_workers: 3
autoscaling_mode: default
head_node:
  InstanceType: m5.large
  ImageId: ami-0abcdef1234567890
worker_nodes:
  InstanceType: m5.large
  ImageId: ami-0abcdef1234567890
setup_commands:
  - pip install -U "ray[default]"
  - pip install boto3
head_setup_commands:
  - pip install git+https://github.com/my-repo/dcnlp.git@<branch>#egg=dcnlp --user --extra-index-url https://<PAT>:@github.com/
worker_setup_commands:
  - pip install git+https://github.com/my-repo/dcnlp.git@<branch>#egg=dcnlp --user --extra-index-url https://<PAT>:@github.com/
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

