BFF
===

The big friendly filter 😁
(originally written by Dirk G @ AI2)

Getting started
---------------

1. Install Rust on your machine.
    1. `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
    2. Add `~/.cargo/bin` to your `PATH` environment variable.
2. Run `cargo build --release`. It places the binary at `target/release/bff`.
3. Run `./target/release/bff --help` to see the available options.


Examples
--------
The main choices for operation of BFF is the _type_ of deduplication (document-level, paragraph-level, etc) performed. This is controlled with the argument `remove-type` argument. For the usage as described in the DCLM paper, you want to use the `--remove-type naive-both` flag to do both paragraph and document-level removal.

An example run of BFF as described in the main paper, where the inputs are located in `/data/inputs` and outputs are to be placed in `/data/outputs`:
```
cargo run --release bff \
   --inputs /data/inputs \
   --output-directory /data/outputs \
   --expected-ngram-count <NGRAM COUNT HERE> \
   --fp-rate 0.01 \
   --min-ngram-size 13 \
   --max-ngram-size 13 \
   --filtering-threshold 0.8 \
   --remove-type old-both \
   --annotate 
```


Usage Notes
--------
See the notes in `/dedup/README.md` for some usage notes. For specific BFF notes:

- The expected ngram count need not be super accurate. Overestimates are better, but minor miscalculations here only affect false positives, which should be quite low with the speficied parameters above.
- Parallelism for large datasets is done by splitting the dataset into "shards" and deduplicating each shard separately. This is controlled with the `shard-num` and `total-shards` arguments
- To get a sense of how much RAM is required to run BFF for a specific dataset and false-positive rate, one can run the sysreq command:
``` 
cargo run --release sysreq \
   --expected-ngram-count <NGRAM COUNT HERE> \
   --fp-rate <FP RATE HERE>
```
