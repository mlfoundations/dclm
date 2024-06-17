# In-depth Descriptions of Mappers, Filters, and Modifiers

## Table of Contents
1. [Introduction](#introduction)
2. [Key Concepts for Processing](#key-concepts-for-processing)
3. [Example YAML Configuration for Processing](#example-yaml-configuration-for-processing)
4. [Running the Processing Pipeline](#running-the-processing-pipeline)
5. [Setting Up a Ray Cluster](#setting-up-a-ray-cluster)

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
