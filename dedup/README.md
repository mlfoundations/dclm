Deduplication Tools
===================


The following subdirectories contain standalone rust projects for running various flavours of deduplication. Within each is a README that includes details and instructions for usage. 

Each of these projects lives in its own separate (public) github repository, but we deem it far simpler to just copy the source code here than to use git submodules. However, we will link the original repositories for reference:

| Subdirectory         | Function     | Original Repository |
|--------------|-----------|------------|
| bff | Bloom-filter deduplication      | [revbucket/bff:ai2-fuzzy-substr\*](https://github.com/revbucket/bff/tree/ai2-fuzzy-substr/)       |
| minhash-rs      | MinHash fuzzy document-level deduplication | [revbucket/minhash-rs:uf](https://github.com/revbucket/minhash-rs/tree/uf)       |
| exact-dedup-rs     | Exact document-level deduplication| [revbucket/rust-exact-dedup](https://github.com/revbucket/rust-exact-dedup)       |

\*We are exceptionally grateful for the help of AI2 with the development of BFF, which we have modified with a few extra tricks for this project. We will respond to bugs/issues (which are probably our fault), but our code here is built on top of the original [repository](https://github.com/allenai/bff).


General Usage Notes
-------------------
- Each of these tools is built in Rust and is intended to be run in a single-node setting. These are all designed to be single-node parallelizable, using multiple local CPUs, but multi-node parallelism is not supported out of the box. Our recommendation in this case is to manually manage the multi-node parallelism. Specific parameters to be amenable to multi-node parallelism are provided in the individual README's in the subdirectories. 
- S3 or other cloud-based storage options can be slightly finicky in rust, often causing hard-to-debug errors. Our recommendation is to download whichever pools to a local disk and point the outputs to a local disk. We've found that [s5cmd](https://github.com/peak/s5cmd) is the best solution for downloading large datasets from S3. 
- For use on AWS, in many cases, we've found the the i4i instances provide the best bang for the buck with these tools, with the AWS Nitro drives (and a raid0 array for larger instances).

