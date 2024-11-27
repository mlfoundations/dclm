Developed by Matt Jordan [revbucket](https://github.com/revbucket/) 2024 

# Tokenize/Shuffle

Once a dataset of properly preprocessed .jsonl(.zst/.zstd/.gz) files have been generated, the final step is to tokenize every document, break the tokens into contexts of appropriate length, and then shuffle them a priori.

This tool, built in rust, operates a few orders of magnitude faster than the tooling built in Ray and only requires a single node.

## Method (algorithmics):
Here's how the tokenization and context generation works.
For a **single** .jsonl file, will loop over all jsons within it and:
1. Tokenize the `text` field and add it to a list of tokens, with an `EOT` token to indicate the end of a single document.
2. Once that token list has size at least `seqlen`, the first `seqlens` will get saved as a context.
If there are any tokens left after parsing the entire file, `PAD` tokens will be added so one final context can be created. 

Once all contexts have been generated, these are randomly shuffled. These shuffled contexts are then grouped into chunks of size `wds_chunk_size` and saved as tar files. Finally a `manifest.json` is generated which lists all tarfiles and how many contexts they each contain.


## Method (engineering):
Since shuffling a list of contexts generally requires that list be able to fit into RAM, we perform a two-stage shuffle which ultimately yields a randomly shuffled collection of contexts ("randomly shuffled" here meaning the order is sampled uniformly from the set of permutations). The two-stage shuffle unburdens the RAM usage but requires the dataset be able to fit on a local disk.

In stage 1 of the shuffle, each context is randomly appended onto a "local cell" somewhere on the local cell directory. The number of "local cells" we initiate should be such that: i) several chunks worth of contexts can fit within one local cell, ii) each individual local cell can be loaded into RAM. Once all local cells are populated, each local cell contains a random choice of contexts, but since these were written in an append-only order, are not shuffled. 

Stage 2 involves a final shuffle step. Each local cell is loaded, and all contexts within that cell are shuffled. Chunks of size `wds_chunk_size` are generated from this now-shuffled list of contexts. Any leftovers are added to an overflow pool that gets handled in a similar fashion to step 2, once all local cells have been generated.

## System Requirements: 
The purpose of the two-step shuffling is to offload the memory to disk. What will typically be required here is a machine that has:
- enough RAM to process `num_threads * num_local_cells` local cells. Typically this will be much much smaller than the size of the dataset itself.
- enough memory to hold the entire dataset

## Examples: 
Make sure Rust is installed on your system: 
    1. `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
    2. Add `~/.cargo/bin` to your `PATH` environment variable.
Construct a directory on a device with enough storage to hold the entire dataset once tokenized. We find that AWS Nitro Drives on EC2 instances are quite useful here (but maybe use raid0 for virtually merging several Nitro drives).

Then run the main rust tool:
```
cargo run --release -- \
--input path/to/raw/dataset \
--local-cell-dir path/to/storage/for/local/cells \ 
--output path/to/output/location \
--tokenizer "EleutherAI/gpt-neox-20b" \ #other supported option is "meta-llama/Meta-Llama-3-8B"
--seqlen 2049 \
--wds-chunk-size 8192 \
--num-local-cells 512 \ # 512 is a good compromise, but might need to raise this much higher for really large datasets \
```


## Usage Notes: 
Some usage notes:
- We support both huggingface's Tokenizers library and TikToken. TikToken is much much faster and has a significantly lower memory footprint, but has some weird behavior regarding repeated whitespace characters with TikToken+`"EleutherAI/gpt-neox-20b"`
- You may run into an error with too many open files. Running `ulimit -Sn 1000000` will fix this
- We support reading the raw data from s3, but the local cells need to live on disk
- If running on EC2, we recommend a c6a/c6g/i4i type instance. Here's a helpful script to install all necessary packages: 
```
sudo yum install git -y 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
bash rustup.sh -y
source ~/.bashrc
git clone <THIS REPO> # CHANGE IF USING FORKED REPO
cd tokshuf-rust # OR WHEREVER THIS TOOL ENDS UP LIVING
sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install g++ -y
aws configure set aws_access_key_id [REDACTED: FILL IN WITH YOUR DATA]
aws configure set aws_secret_access_key [REDACTED: FILL IN WITH YOUR DATA]
aws configure set default.region [REDACTED: FILL IN WITH YOUR DATA]
cargo build --release 
# And then run as above^^^
```
