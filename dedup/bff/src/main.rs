use ahash::RandomState;
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use clap::{Parser, Subcommand};
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use serde_json::Value;
use std::clone::Clone;
use std::collections::VecDeque;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::cmp;
use std::io::{BufRead, BufReader, BufWriter, Write, Cursor};
use std::mem::size_of;
use std::path::{PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread::available_parallelism;
use threadpool::ThreadPool;
use unicode_segmentation::UnicodeSegmentation;
use rayon::prelude::*;
use sysinfo::{
    System,
};
use glob::glob;
use human_bytes::human_bytes;
use std::fs::{OpenOptions, create_dir_all};
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar,ProgressStyle};
use std::time::{Instant};

use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_s3::{Client};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::operation::get_object::GetObjectOutput;
use tokio::io::AsyncReadExt;
use tokio::io::BufReader as tBufReader;
use tokio::time::{Duration, sleep};
use async_compression::tokio::bufread::GzipDecoder as asyncGZ;
use async_compression::tokio::bufread::ZstdDecoder as asyncZstd;
use zstd::stream::write::Encoder as ZstdEncoder;
use zstd::stream::read::Decoder as ZstDecoder;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(arg_required_else_help = true)]
    Bff {
        /// (List of) directories or files that are jsonl.gz files
        #[arg(required=true, long)]
        inputs: Vec<PathBuf>,

        /// Output directory where the deduplicated files will end up. 
        /// These will have the same basename as the inputs, so it is up to you to ensure no collisions here!
        #[arg(required=true, long)]
        output_directory: PathBuf,    

        /// If specified, tries to load the bloom filter from this file, and will save once complete.
        /// If unspecified, will not save the bloom filter at the end    
        #[arg(long)]
        bloom_filter_file: Option<PathBuf>,

        /// The number of expected ngrams. This is used to calculate the optimal number of hashers.
        /// If the filter already exists, this parameter is ignored.
        #[arg(required=true, long)]
        expected_ngram_count: usize,

        /// The desired false positive rate
        /// Note that this is a per-ngram FP rate, and not a per-paragraph rate
        #[arg(required=true, long)]
        fp_rate: f64,    

        /// The smallest ngram size to consider. Paragraphs that have fewer than this number of tokens
        /// are not deduplicated and always kept. These ngrams are never added to the bloom filter.
        /// Note that this value only matters if the paragraph has fewer tokens than the max ngram size.
        #[arg(long, default_value_t = 20)]
        min_ngram_size: usize,

        /// The largest ngram size to consider. Paragraphs are deduplicated based on the number of
        /// ngrams of this size that are already present in the bloom filter.
        #[arg(long, default_value_t = 20)]
        max_ngram_size: usize,

        /// If this fraction of ngrams of the max ngram size are already present in the bloom filter,
        /// the paragraph is considered a duplicate and is discarded.
        /// Set this to 0 to never produce any output. This is useful when you want to prime the filter
        /// with some content that should be considered duplicates, without deduplicating that content
        /// itself.
        #[arg(long, default_value_t = 0.80)]
        filtering_threshold: f64,

        /// How many tokens to count as a duplicate in substring mode
        #[arg(long, default_value_t = 50)]
        substr_seqlen: usize,

        /// Which "BFF mode" we're in. We have options of 'paragraph', 'document', 'both'
        /// indicating we remove individual paragraphs/documents if duplicated
        /// The logic for "both" mode is a bit subtle. See comments below
        #[arg(long, default_value_t = RemoveType::Paragraph, value_enum)]
        remove_type: RemoveType,        


        /// Number of hashers to use in bloom filter 
        /// 0 is the default in which case we use the optimal number 
        #[arg(long, default_value_t = 0)]
        num_hashers: usize,

        /// Whether or not to update the bloom filter. If this is true, the filter is not updated, but
        /// the input is still deduplicated based on the filter. Default is false.
        #[arg(long, default_value_t = false)]
        no_update_bloom_filter: bool,

        /// If this is true, we keep the input intact, but we add an annotation to each document that
        /// explains which spans from the text would have been deleted.
        #[arg(long, default_value_t = false)]
        annotate: bool,

        /// The number of threads to use for processing.
        /// If this is 0, the number of threads is automatically determined.
        #[arg(long, short = 't', default_value_t = 0)]
        threads: usize,

        /// If this flag is present, we will never save a bloom filter to disk
        #[arg(long, default_value_t = false)]
        no_save_bloom_filter: bool,


        /// Turn this flag on if we don't want to use a progress bar 
        /// Helpful when running through ssh and wanting to check progress via logs and not a terminal
        #[arg(long, default_value_t = false)]
        no_progress_bar: bool,

        /// For virtual "sharding", this param and the next one subselect files to deduplicate together
        /// Defaults to no virtual sharding
         #[arg(long, default_value_t=0)]
        shard_num: usize,

        #[arg(long, default_value_t=1)]
        total_shards: usize,   


    },

    Sysreq {
        /// Handy tool to help guess RAM requirements for a given pool of data
        #[arg(required=true, long)]
        expected_ngram_count: usize,
        #[arg(required=true, long)]
        fp_rate: f64,
        #[arg(long, default_value_t=0)]
        num_hashers: usize,
    },
        
}

#[derive(Debug, Clone, Eq, PartialEq, clap::ValueEnum)]
enum RemoveType {
    /// Types for what we check to see if is a duplicate

    ///Checks each paragraph of size >=min_ngram_size if it is duplicated. If so, it gets removed
    Paragraph,

    /// Checks if enough of the ngrams of size ==max_ngram_size (or just one ngram if tokens in range [min_ngram_size, max_ngram_size])
    /// and if enough are present in filter, the whole document gets removed 
    Document,  

    /// Does paragraph removal, BUT if enough of the paragraph ngram checks are contained, removes the whole document
    Both,     

    /// Does Both like above, but does it in the naive way by doing doc level first and then paragraph level
    /// Warning: There are some SUBTLE differences between the output of RemoveType::Both and RemoveType::NaiveBoth, 
    NaiveBoth,

    /// Both as we did it in dabv3
    OldBoth,

    /// Does substring style removal
    Substring,

    /// Does exact removal (it's not _really_ exact, shhh....)
    Exact,


}


fn not_whitespace(w: &str) -> bool {
    for c in w.chars() {
        if !c.is_whitespace() {
            return true;
        }
    }
    false
}


fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|w| {not_whitespace(w)})
}

fn tokenize_indices(s: &str) -> impl Iterator<Item = (usize, &str)> {
    s.split_word_bound_indices().filter(|(_, w)| {not_whitespace(w)})
}


fn merge_intervals(mut v: Vec<(usize, usize)>, already_sorted: bool) -> Vec<(usize, usize)>{    
    if !already_sorted {
        v.sort_by_key(|(key, _)| key.clone());
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in v {
        if merged.len() == 0 {
            merged.push((s, e));
        } else if merged.last().unwrap().1 >= s {
            let (old_s, old_e) = merged.pop().unwrap();
            merged.push((old_s, cmp::max(e, old_e)));
        } else {
            merged.push((s, e));
        }
    }
    merged
}

fn merge_sorted_interval_pair(u: Vec<(usize, usize)>, w: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    // Given two sorted lists of intervals, does a merge of the pairs, and then unions all intervals
    let mut v : Vec<(usize, usize)> = Vec::new();
    let mut ui = 0;
    let mut wi = 0;
    while ui < u.len() && wi < w.len() {
        let (us, ue) = u[ui];
        let (ws, we) = w[wi];
        if us < ws || (us == ws && ue <= we){
            v.push((us, ue));
            ui += 1;
        } else {
            v.push((ws, we));
            wi += 1
        } 
    }
    while ui < u.len() {
        v.push(u[ui]);
        ui += 1;
    }

    while wi < w.len() {
        v.push(w[wi]);
        wi += 1;
    }

    merge_intervals(v, true)
}



fn invert_intervals(mut v: Vec<(usize, usize)>, already_sorted: bool, max_len: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    // Returns both (regular, inverted) intervals 
    // (inverted intervals are all the "gaps")
    // i.e., for a list of range 10, the inverse of [(0,4) (6, 8)] -> [(4,6), (8,10)]
    if !already_sorted {
        v.sort_by_key(|(key, _)| key.clone());
    }
    let mut inv_intervals : Vec<(usize, usize)> = Vec::new();
    if v.len() == 0 {
        inv_intervals.push((0,  max_len));
        return (v, inv_intervals);
    }

    if v.first().unwrap().0 > 0 {
        inv_intervals.push((0, v.first().unwrap().0));
    }
    for window in v.windows(2) {
        let (_, e0) = window[0];
        let (s1, _) = window[1];
        inv_intervals.push((e0, s1));
    }
    if v.last().unwrap().1 < max_len {
        inv_intervals.push((v.last().unwrap().1, max_len));
    }
    (v, inv_intervals)
}


fn fuzzy_sandwich_intervals(v: &Vec<(usize, usize)>, foward: bool, threshold: f64) -> Vec<(usize, usize)> {
    // Given SORTED list of DISJOINT intervals, scans in the forward/!forward direction
    // And collects all intervals that: 
    // 1. Start and end at an interval
    // 2. Have >=threshold of the range contained in an input interval
    // e.g. [(0,9), (10, 20)] -> [(0,20)] (when the threshold is <=0.95)

    let n = v.len();
    let iter_range : Vec<_> = if foward {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };
    let mut output : Vec<(i32, i32, i32)> = Vec::new();
    for idx in iter_range {

        let (next_s, next_e) = v[idx];
        let next_s = next_s as i32;
        let next_e = next_e as i32;

        if output.len() == 0 {
            output.push((next_s, next_e, next_e - next_s));
            continue;
        }
        let (cur_s, cur_e, cur_w) = output.last().unwrap();
        let new_interval = (cmp::min(next_s, *cur_s as i32), 
                            cmp::max(next_e, *cur_e as i32), 
                            *cur_w  as i32 + next_e - next_s);
        if new_interval.2 as f64 >= (new_interval.1 - new_interval.0) as f64 * threshold {
            output.pop().unwrap();
            output.push(new_interval);
        } else {
            output.push((next_s, next_e, next_e - next_s));
        }
    }

    output
        .iter()
        .map(|(a,b, _)| (*a as usize, *b as usize))
        .collect()
}


/*=============================================================
=                     Bloom Filter stuff                      =
==============================================================*/

struct BloomFilter {
    bits: Vec<AtomicU32>,
    hash_builder_seeds: Vec<[u64; 4]>, // RandomState does not store its seeds, so we have to store them ourselves.
    hash_builders: Vec<RandomState>,
}

impl BloomFilter {
    const MAGIC: u32 = 0x81F0_F117;
    const VERSION: u32 = 1;

    fn optimal_number_of_hashers(size_in_bytes: usize, expected_elements: usize) -> usize {
        let expected_elements = expected_elements as f64;
        let size_in_bits = (size_in_bytes * 8) as f64;
        let k = (size_in_bits / expected_elements) * (2.0f64.ln());
        k.ceil() as usize
    }

    fn prob_of_false_positive(
        size_in_bytes: usize,
        expected_elements: usize,
        num_hashers: usize,
    ) -> f64 {
        let k = num_hashers as f64;
        let m = (size_in_bytes * 8) as f64;
        let n = expected_elements as f64;
        (1.0 - (1.0 - (1.0 / m)).powf(k * n)).powf(k)
    }


    fn my_prob_of_false_positive(&self, expected_elements: usize) -> f64 {
        Self::prob_of_false_positive(
            self.size_in_bytes(),
            expected_elements,
            self.hash_builders.len(),
        )
    }

    fn size_in_bytes(&self) -> usize {
        self.bits.len() * size_of::<AtomicU32>()
    }

    fn calculate_sparsity(&self) -> f64 {
        let set_bits:usize = self.bits.par_iter()
            .map(|atomic| {
                let value = atomic.load(std::sync::atomic::Ordering::Relaxed);
                value.count_ones() as usize
            })
            .sum();
        let total_bits = self.size_in_bytes() * 8;
        return (set_bits as f64) / (total_bits as f64);
    }

    fn new(size_in_bytes: usize, num_hashers: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers);
        let mut hash_builders = Vec::with_capacity(num_hashers);
        for _ in 0..num_hashers {
            let seeds = rng.gen::<[u64; 4]>();
            hash_builders.push(RandomState::with_seeds(
                seeds[0], seeds[1], seeds[2], seeds[3],
            ));
            hash_builder_seeds.push(seeds);
        }

        let number_of_u32 = size_in_bytes / size_of::<AtomicU32>();
        let bits = {
            (0..number_of_u32).into_par_iter().map(|_| AtomicU32::default()).collect()
        };


        Self {
            bits,
            hash_builder_seeds,
            hash_builders,
        }
    }



    fn from_file(path: &PathBuf) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(path)?;
        let mut stream = BufReader::new(&mut file);

        let magic: u32 = stream.read_u32::<LittleEndian>()?;
        if magic != Self::MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
        }

        let version: u32 = stream.read_u32::<LittleEndian>()?;
        if version != Self::VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid version",
            ));
        }

        let num_hashers: u32 = stream.read_u32::<LittleEndian>()?;
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers as usize);
        let mut hash_builders = Vec::with_capacity(num_hashers as usize);
        for _ in 0..num_hashers {
            let seeds = [
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
            ];
            hash_builders.push(RandomState::with_seeds(
                seeds[0], seeds[1], seeds[2], seeds[3],
            ));
            hash_builder_seeds.push(seeds);
        }

        let number_of_elements = stream.read_u64::<LittleEndian>()?;
        let mut bits = Vec::new();
        bits.reserve_exact(number_of_elements as usize);
        for _ in 0..number_of_elements {
            bits.push(AtomicU32::new(stream.read_u32::<NativeEndian>()?));
        }

        Ok(Self {
            bits,
            hash_builder_seeds,
            hash_builders,
        })
    }

    fn write_to_file(&self, path: &PathBuf) -> io::Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let mut stream = BufWriter::new(&file);

        stream.write_u32::<LittleEndian>(Self::MAGIC)?;
        stream.write_u32::<LittleEndian>(Self::VERSION)?;
        stream.write_u32::<LittleEndian>(self.hash_builder_seeds.len() as u32)?;
        for hash_builder_seed in &self.hash_builder_seeds {
            for seed in hash_builder_seed {
                stream.write_u64::<LittleEndian>(*seed)?;
            }
        }

        stream.write_u64::<LittleEndian>(self.bits.len() as u64)?;
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                self.bits.as_ptr().cast::<u8>(),
                self.bits.len() * size_of::<AtomicU32>(),
            );
            stream.write_all(bytes)?;
        };

        Ok(())
    }

    fn hashes(&self, s: &VecDeque<&str>) -> Vec<u64> {
        self.hash_builders
            .iter()
            .map(|hash_builder| {
                let mut hasher = hash_builder.build_hasher();
                s.hash(&mut hasher);
                hasher.finish()
            })
            .collect()
    }

    fn insert_hashes(&self, hashes: &Vec<u64>) {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            self.bits[index].fetch_or(1 << bit, Ordering::Relaxed);
        }
    }

    #[allow(dead_code)] // use in unit test
    fn insert(&self, s: &VecDeque<&str>) {
        let hashes = self.hashes(s);
        self.insert_hashes(&hashes);
    }

    fn contains_hashes(&self, hashes: &Vec<u64>) -> bool {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            if self.bits[index].load(Ordering::Relaxed) & (1 << bit) == 0 {
                return false;
            }
        }
        true
    }

    #[allow(dead_code)] // use in unit test
    fn contains(&self, s: &VecDeque<&str>) -> bool {
        let hashes = self.hashes(s);
        self.contains_hashes(&hashes)
    }


    fn from_args(bloom_filter_file: Option<PathBuf>, expected_ngram_count: usize, fp_rate: f64, num_hashers: usize) -> Self {
        /* Uses the CLI args to build a bloom filter 
        Logic:
            - Check if file exists, if so, just load it and return 
            - Get size:
                + if size is explicitly speciifed, use this
                + otherwise, compute based on ngrams + fp rate 
            - Return 
        */        

        let bloom_filter = match &bloom_filter_file {
            Some(path) if path.exists() => {
                println!("Loading bloom filter from {:?}...", path);
                BloomFilter::from_file(&path).unwrap()
            }
            _ => {
            println!("Creating new bloom filter...");
            let bloom_filter_size = compute_bloom_size(fp_rate, expected_ngram_count, true, num_hashers);
            let num_hashers = BloomFilter::optimal_number_of_hashers(
                bloom_filter_size,
                expected_ngram_count,
            );
            BloomFilter::new(bloom_filter_size, num_hashers)
            }
        };

        println!("Bloom filter has size {} | FP Rate {:?}",
                 human_bytes(bloom_filter.size_in_bytes() as f64), 
                 bloom_filter.my_prob_of_false_positive(expected_ngram_count));
        bloom_filter
    }

}



fn compute_bloom_size(fp_rate: f64, expected_ngram_count: usize, limit_to_sys: bool, num_hashers: usize) -> usize {
    /* Uses binary search to find optimal size of bloom filter using optimal number of hashers
       and provided ngram counts
    */
    // compute 90% of system ram 
    let mut sys = System::new_all();
    sys.refresh_all();


    let mut lo = 1 as usize;

    let mut hi = if limit_to_sys {
                    ((sys.total_memory() as f64) * 0.9) as usize
                 } else {
                    420_744_073_709_551_615 as usize
                 };

    let compute_hashers = num_hashers == 0;

    // Save some time by checking endpoint first
    let num_hashers = if compute_hashers {
         BloomFilter::optimal_number_of_hashers(hi, expected_ngram_count)
    } else {
        num_hashers
    };

    if limit_to_sys && BloomFilter::prob_of_false_positive(hi, expected_ngram_count, num_hashers) > fp_rate {
        println!(
            "WARNING: To achieve desired false-positive rate, you'd need >90% of system RAM. Defaulting to 90% \
            system RAM.");
        return hi;
    }

    // Then do binary search to find optimal size
    while lo < hi-1 {
        let mid = lo + (hi - lo) / 2;
        let num_hashers = if compute_hashers {
            BloomFilter::optimal_number_of_hashers(mid, expected_ngram_count)
        } else {num_hashers};
        let computed_fp = BloomFilter::prob_of_false_positive(mid, expected_ngram_count, num_hashers) ;
        if computed_fp > fp_rate {
            // FP rate too high, need to go bigger
            lo =  mid + 1;
        } else {
            // FP rate too low, can make bloom filter smaller
            hi = mid -1;
        }
    }
    hi
}





#[allow(clippy::too_many_arguments)] // TODO : abstract parameters into a struct
async fn process_file(
    input_file: &PathBuf,
    output_file: &PathBuf,
    bloom_filter: &Arc<BloomFilter>,
    max_ngram_size: usize,
    min_ngram_size: usize,
    substr_seqlen: usize,
    remove_type: &RemoveType,
    filtering_threshold: f64,
    no_update_bloom_filter: bool,    
    annotate: bool,
    pbar_option: &Option<Arc<Mutex<ProgressBar>>>,    
) -> Result<(usize, usize), io::Error> {

    // Setup input/output writers
    // If input file is local: can stream pretty easily/robustly
    // If input file is s3: load entire file and split it into thing that implements .lines() iterator
    let lines : Box<dyn Iterator<Item = Result<String, std::io::Error>>> = if is_s3(input_file) {
        /*
        let input_file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(input_file)?;
        BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(input_file)).lines()        
        */
        Box::new(get_reader_from_s3(input_file, None).await.unwrap().lines())
    } else {
        let ext = input_file.extension().unwrap().to_str().unwrap();
        let input_file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(input_file)?;

        match ext {
            "zstd" | "zst" => {
                Box::new(BufReader::with_capacity(1024 * 1024, ZstDecoder::new(input_file).unwrap()).lines())
            }, 
            "gz" => {
                Box::new(BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(input_file)).lines())                
            }
            _ => {
                Box::new(BufReader::with_capacity(1024 * 1024, input_file).lines()) 
            }
        }
    };

    // If output file is local, write directly to file
    let mut output_data: Vec<u8> = Vec::new();


    // Loop over lines and do bff stuff
    let mut count = 0;
    let mut fully_skipped = 0;
    let mut removed_text_bytes = 0;
    let mut total_text_bytes = 0;
    for line in lines {
        let line = line?;
        count += 1;
        let (dedup_data, removed_line_bytes, total_line_bytes) = match *remove_type {
            RemoveType::Exact => {
                process_line_exact(&line, &bloom_filter, no_update_bloom_filter, annotate)
            }
            RemoveType::Substring => {
                process_line_substring(&line, &bloom_filter, max_ngram_size,
                                   no_update_bloom_filter, annotate, substr_seqlen, filtering_threshold)
            }
            RemoveType::Both => {
                // Dumb version: check if document should be removed
                process_line_both(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                                  filtering_threshold, no_update_bloom_filter, annotate)
                /*
                let (doc_dedup, doc_removed_bytes, doc_total_bytes) = 
                    process_line(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                         &RemoveType::Document, filtering_threshold, no_update_bloom_filter, annotate);
                if doc_removed_bytes > 0 { 
                    (doc_dedup, doc_removed_bytes, doc_total_bytes)
                } else { // and if document should NOT be removed, then do paragraph level
                    process_line(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                         &RemoveType::Paragraph, filtering_threshold, no_update_bloom_filter, annotate)     
                } 
                */
                    
            },
            RemoveType::OldBoth => {
                process_line_oldboth(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                                     filtering_threshold, no_update_bloom_filter, annotate)
            }


            RemoveType::NaiveBoth => {
                let (doc_dedup, doc_removed_bytes, doc_total_bytes) = 
                    process_line(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                         &RemoveType::Document, filtering_threshold, no_update_bloom_filter, annotate);
                if doc_removed_bytes > 0 { 
                    (doc_dedup, doc_removed_bytes, doc_total_bytes)
                } else { // and if document should NOT be removed, then do paragraph level
                    process_line(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                         &RemoveType::Paragraph, filtering_threshold, no_update_bloom_filter, annotate)     
                }                 
            }
            _ => { // Handles the "paragraph" and "document" case (but not "both" [for now]!)
                    process_line(&line, &bloom_filter, min_ngram_size, max_ngram_size,
                         remove_type, filtering_threshold, no_update_bloom_filter, annotate)             
            }
        };

        removed_text_bytes += removed_line_bytes;
        total_text_bytes += total_line_bytes;
        if dedup_data.get("text").unwrap().as_str().unwrap().trim().is_empty() {
            fully_skipped += 1
        }
        else {
            output_data.extend(serde_json::to_vec(&dedup_data).unwrap());
            output_data.extend(b"\n");         
        }        
    }

    // Handle output files
    let output_data = compress_data(output_data, &output_file);
    if fully_skipped < count {
        if is_s3(output_file) {
            let (output_bucket, output_key) = split_s3_path(output_file);
            let client = get_s3_client().await;
            let _ = client
                .put_object()
                .bucket(output_bucket)
                .key(output_key)
                .body(ByteStream::from(output_data))
                .send()
                .await;            
        } else {
            let mut output_file = OpenOptions::new()
                .read(false)
                .write(true)
                .create(true)
                .truncate(true)
                .open(output_file)?;            
            output_file.write_all(&output_data)?;
            
        }
    }


    match pbar_option {
        Some(pbar) => {
            let pb = pbar.lock().unwrap();
            pb.inc(1);
        }
        None => (),
    }
    Ok((removed_text_bytes, total_text_bytes))
}


fn process_line(line: &String, bloom_filter: &BloomFilter, min_ngram_size: usize, max_ngram_size: usize,
               remove_type: &RemoveType, filtering_threshold: f64, no_update_bloom_filter: bool, annotate: bool) ->  
    (serde_json::Value, usize, usize) {
    // Main BFF logic: processes a single json document
    // Does the following (handling the {paragraph, document, both} cases)
    // 1. Breaks document into units (paragraph/both -> paragraph; document -> full text)
    // 2. For each unit, tokenize and 
    //    a. if num_tokens < min_ngram_size: do nothing, leave this unit intact
    //    b. if num_tokens >= max_ngram_size: break unit into ngram-shingling of max_ngram_size
    //    c. else, full unit is treated as one ngram
    // 3. Check containment of each ngram in bloom filter.
    //    a. If > filtering_threshold contained, mark unit for deletion
    // 4. If unit survives step 3, add all ngrams into bloom filter 
    // 5. BOTH-mode ONLY: If total_contained_ngrams * threshold >= total_ngrams, omit the WHOLE document

    // Outputs are (output_json, total_removed_bytes, total_input_bytes)
    // If annotate is turned on, nothing gets removed, text is left intact, but byte-windows-removed


    let mut data: Value = serde_json::from_str(&line).unwrap();
    let mut total_bytes = 0;
    let mut removed_bytes = 0;
    let text = data["text"].as_str().unwrap();

    // Step 1: Break text into "units"
    let newlines = if *remove_type == RemoveType::Document {
        vec![0, text.len()]
    } else {
        let mut newlines = Vec::new();
        newlines.push(0);
        for i in text.match_indices('\n') {
            newlines.push(i.0);
        }
        newlines.push(text.len());
        newlines
    };
    let mut windows_to_remove = Vec::new();


    let mut total_ngrams = 0;
    let mut total_contained_ngrams = 0;
    for paragraph_window in newlines.windows(2) {
        let paragraph = &text[paragraph_window[0]..paragraph_window[1]];
        total_bytes += paragraph.len();
        

        // Step 2: Tokenize and chunk into ngram shinglings, hash each one for the bff
        let mut hashes: Vec<Vec<u64>> = Vec::new();
        let mut ngram: VecDeque<&str> = VecDeque::with_capacity(max_ngram_size);
        for token in tokenize(paragraph) {
            ngram.push_back(token);
            if ngram.len() >= max_ngram_size { // Step 2b: ngram shingling long enough
                hashes.push(bloom_filter.hashes(&ngram));
                ngram.pop_front();
            }
        }
        // Step 2c: unit is short, but not TOO SHORT
        if hashes.is_empty() && ngram.len() >= min_ngram_size {
            hashes.push(bloom_filter.hashes(&ngram));
        }


        // Step 3: check containment of ngrams
        let contained_ngrams = hashes
            .iter()
            .filter(|ngram| bloom_filter.contains_hashes(ngram))
            .count();
        total_ngrams += hashes.len();
        total_contained_ngrams += contained_ngrams;        
        let number_of_ngrams = hashes.len() as f64;
        //windows_to_remove.ansoteuhoausenh();
        let should_remove = contained_ngrams as f64 / number_of_ngrams > filtering_threshold;
        if  should_remove {
            windows_to_remove.push(paragraph_window);
            removed_bytes += paragraph.len();
        } else if !no_update_bloom_filter { 
            // Step 4: add all ngrams to the bloom filter if we don't remove it
            for ngram in hashes {
                bloom_filter.insert_hashes(&ngram);
            }
        }
    }

    // Step 5: Handle the both case
    let temp_window = vec![0, text.len()];
    if *remove_type == RemoveType::Both && 
        (total_contained_ngrams as f64) / (total_ngrams as f64) > filtering_threshold {
        windows_to_remove.clear();
        windows_to_remove.push(&temp_window);
    }

    // Format outputs: 
    if annotate {
        data["bff_duplicate_spans"] = serde_json::to_value(windows_to_remove).unwrap();
        data["bff_contained_ngram_count"] = serde_json::to_value(total_contained_ngrams).unwrap();
    } else {
        let mut output_paragraphs = String::new();
        let mut last_end = 0;
        for paragraph_window in windows_to_remove {
            output_paragraphs.push_str(&text[last_end..paragraph_window[0]]);
            last_end = paragraph_window[1];
        }
        output_paragraphs.push_str(&text[last_end..]);
        data["text"] = Value::String(output_paragraphs);
    }

    (data, removed_bytes, total_bytes)
}


fn process_line_substring(line: &String, bloom_filter: &BloomFilter, max_ngram_size: usize,
                           no_update_bloom_filter: bool, annotate: bool, substr_seqlen: usize, 
                           fuzzy_threshold: f64) -> 
    (serde_json::Value, usize, usize) {
    let mut data: Value = serde_json::from_str(&line).unwrap();
    let text = data["text"].as_str().unwrap();
    let mut total_tokens: usize = 0;
    let total_bytes = text.len();

    // Step 1: Get contained ngram indices, and map from ngram/token idx -> text idx
    let mut hashes : Vec<Vec<u64>> = Vec::new(); // Note: hashes[i] is the hash of tokens[i..i + max_ngram_size]
    let mut ngram: VecDeque<&str> = VecDeque::with_capacity(max_ngram_size); // temp thing to get hashes
    let mut tokenidx2textidx: Vec<usize> = Vec::new(); // token_idx -> text idx

    let mut debug_tokens : Vec<&str> = Vec::new();
    for (text_idx, token) in tokenize_indices(text) {
        debug_tokens.push(token);
        total_tokens += 1;
        tokenidx2textidx.push(text_idx);
        ngram.push_back(token);
        if ngram.len() >= max_ngram_size {
            let cur_hash = bloom_filter.hashes(&ngram);
            hashes.push(cur_hash.clone());
            // Wild idea: what if we pushed ALL ngram hashes to bloom filter?
            // bloom_filter.insert_hashes(&cur_hash);
            ngram.pop_front();
        }
    }
    if hashes.len() == 0 { // Too short of document, do nothing, return early
        return (data, 0, total_tokens);
    }
    tokenidx2textidx.push(text.len()); // bookend 
    // UPDATE 
    // Get hash intervals that are contained -- actual intervals, so semiopen like [)
    let mut contained_hash_ranges : Vec<(usize, usize)> = Vec::new();
    let mut contained_ngram_count = 0;
    for (idx, hash) in hashes.iter().enumerate() {
        let contain = bloom_filter.contains_hashes(hash);
        if !contain { continue; }
        contained_ngram_count += 1;
        if contained_hash_ranges.len() > 0 && contained_hash_ranges.last().unwrap().1 == idx {
            let (start, _) = contained_hash_ranges.pop().unwrap();
            contained_hash_ranges.push((start, idx + 1));
        } else {
            contained_hash_ranges.push((idx, idx+1))
        }
    }

    // And then convert hash ranges to token intervals, merge, filter out short tokens, and get to_keep intervals
    let contained_token_intervals : Vec<(usize, usize)> = contained_hash_ranges.iter()
        .map(|(s, e)| (*s, e + max_ngram_size - 1))
        .collect();

    let contained_token_intervals = merge_intervals(contained_token_intervals, true);
    let contained_token_intervals = if fuzzy_threshold < 1.0 {
        // Do fuzzy thing
        let forward_fuzzy = fuzzy_sandwich_intervals(&contained_token_intervals, true, fuzzy_threshold);
        let backward_fuzzy = fuzzy_sandwich_intervals(&contained_token_intervals, false, fuzzy_threshold);
        merge_sorted_interval_pair(forward_fuzzy, backward_fuzzy)
    } else {
        contained_token_intervals
    };

    let contained_token_intervals : Vec<(usize, usize)> = contained_token_intervals
        .into_iter()
        .filter(|(s, e)| e-s >= substr_seqlen)
        .collect();

    let (contained_token_intervals, noncontained_token_intervals) = invert_intervals(contained_token_intervals, true, total_tokens);
    // Add hashes in to bloom filter
    if !no_update_bloom_filter {
        for (s,e) in &noncontained_token_intervals {
            let mut hash_idx = *s;
            let mut ngram_end = s + max_ngram_size;
            while ngram_end <= *e {
                bloom_filter.insert_hashes(&hashes[hash_idx]);
                hash_idx += 1;
                ngram_end += 1;
            }
        }
    }

    // For the noncontained_token_intervals, map to text indices
    let contained_text_intervals: Vec<(usize, usize)> = contained_token_intervals
        .into_iter()
        .map(|(s,e)| (tokenidx2textidx[s], tokenidx2textidx[e]))
        .collect();
    let noncontained_text_intervals: Vec<(usize, usize)> = noncontained_token_intervals
        .into_iter()
        .map(|(s, e)| (tokenidx2textidx[s], tokenidx2textidx[e]))
        .collect();

    let mut output_str = String::new();
    for (s, e) in noncontained_text_intervals {
        output_str.push_str(&text[s..e]);
    }

    if annotate {
        data["bff_duplicate_spans"] = serde_json::to_value(contained_text_intervals).unwrap();
        data["bff_contained_ngram_count"] = serde_json::to_value(contained_ngram_count).unwrap();
    } else {
        data["text"] = Value::String(output_str.trim_end().to_string());    
    }
    (data, total_bytes - output_str.len(), total_bytes as usize)

}




fn process_line_both(line: &String, bloom_filter: &BloomFilter, min_ngram_size: usize, max_ngram_size: usize,
               filtering_threshold: f64, no_update_bloom_filter: bool, annotate: bool) -> 
    (serde_json::Value, usize, usize) {

    /* Actual paragraph+document level deduplication (but quite complicated)
    Here's the plan:
    1. tokenize the string and create ngram shinglings and hash each of them, get containments
        a. store each hash in a separate paragraph-level bucket
        b. store any ngram that spans across >1 paragraph in a separate bucket 
    2. Check document level containment
        a. If enough of the seen ngrams are contained, we remove the whole document

    3. Check paragraphs: for each paragraph...
        a. Check if enough of its ngrams have been seen before. 
        b. If so, mark this as deleted
        c. If not, add to output text, and insert all of its ngram-hashes to the bloom filter
        d. check the ngrams that span across paragraphs. While 
            i. if the earliest ngram ends before the current paragraph starts, it survived, so we add it to the bloom filter
            ii. if the earliest ngram intersects the current paragraph AND we should delete the current paragraph, 
                delete it from the list. Otherwise, leave it and continue the paragraph looper
    4. Process output as usual
    */

    let mut data: Value = serde_json::from_str(&line).unwrap();
    let text = data["text"].as_str().unwrap();
    let total_bytes = text.len();

    let mut newlines = Vec::new();
    newlines.push(0);
    for i in text.match_indices('\n') {
        newlines.push(i.0)
    }
    newlines.push(text.len());
    let num_paragraphs_plus1 = newlines.len();


    // Some data structures:
    // Map paragraph_id -> 
    //                     all hashes for that paragraph
    //                     num_ngrams
    //                     contained_ngrams

    let mut total_ngram_hashes: Vec<Vec<Vec<u64>>> = vec![Vec::new(); num_paragraphs_plus1];
    let mut total_ngram_counts: Vec<usize> = vec![0; num_paragraphs_plus1];
    let mut contained_ngram_counts: Vec<usize> = vec![0; num_paragraphs_plus1];

    let mut overflow_idxs: Vec<(usize, usize)> = Vec::new(); 
    // ^contains (first_tok.text_idx, last_tok.text_idx) for each ngram in last_idx of totals [overflow idx]


    let mut ngram: VecDeque<&str> = VecDeque::with_capacity(max_ngram_size);
    let mut ngram_idxs: VecDeque<usize> = VecDeque::with_capacity(max_ngram_size);
    let mut paragraph_idx = 0;
    let mut seen_one_ngram = false;
    for (text_idx, token) in tokenize_indices(text) {
        ngram.push_back(token);
        ngram_idxs.push_back(text_idx);

        if ngram.len() >= max_ngram_size {
            // ngram is long enough! Compute hashes, containment, and which paragraph (if any)
            seen_one_ngram = true;
            let hashes = bloom_filter.hashes(&ngram);
            let contains = bloom_filter.contains_hashes(&hashes);
            let paragraph_idx_end = newlines[paragraph_idx + 1];
            let cur_paragraph = if text_idx < paragraph_idx_end {
                paragraph_idx
            } else {
                overflow_idxs.push((*ngram_idxs.front().unwrap(), text_idx));
                num_paragraphs_plus1 - 1 // not fully in any single para => last index                
            };

            // Adjust state for this ngram for its corresponding paragraph
            total_ngram_counts[cur_paragraph] += 1;
            total_ngram_hashes[cur_paragraph].push(hashes);
            contained_ngram_counts[cur_paragraph] += contains as usize;
            ngram.pop_front();
            ngram_idxs.pop_front();

            // Adjust paragraph index accordingly
            let front_idx = ngram_idxs.front().unwrap();
            while *front_idx < newlines[paragraph_idx] {
                paragraph_idx += 1;
            }
        }
    }
    // Handle special case if full string is too small
    if min_ngram_size <= ngram.len() && !seen_one_ngram {
        // Short document, only one ngram, goes in the last index, basically a 'document' removal case
        let hashes = bloom_filter.hashes(&ngram);
        let contains = bloom_filter.contains_hashes(&hashes);
        let cur_paragraph = num_paragraphs_plus1 - 1;

        total_ngram_hashes[cur_paragraph].push(hashes);
        total_ngram_counts[cur_paragraph] += 1;
        contained_ngram_counts[cur_paragraph] += contains as usize;
    }

    // Now check for containment of document
    let total_ngrams: usize = total_ngram_counts.iter().sum::<usize>();
    let total_contains: usize = contained_ngram_counts.iter().sum::<usize>();
    
    if total_ngrams == 0 { // No ngrams, doc was too short, return defaults
        return (data, 0, total_bytes as usize);
    }

    if (total_contains as f64) / total_ngrams as f64 >= filtering_threshold { // remove whole doc
        if annotate {
            let dup_span: Vec<Vec<usize>> = vec![vec![0, total_bytes]];
            data["bff_duplicate_spans"] = serde_json::to_value(dup_span).unwrap();
            data["bff_contained_ngram_count"] = serde_json::to_value(total_contains).unwrap();
        }  else {
            data["text"] = Value::String(String::new());
        }
        return (data, total_bytes, total_bytes)
    }

    // Okay, now do a clever interleaved loop over the paragraphs
    // At the end want to:
    // 1. Build up new text 
    // 2. Build up removal spans 
    // 3. Add hashes ONLY of what survived

    // Trick here is to loop over paragraphs and keep track of 'survivable' hashes in the overflow
    overflow_idxs.reverse();    
    let mut overflow_hashes = total_ngram_hashes[total_ngram_hashes.len() - 1].clone();
    overflow_hashes.reverse();
    let mut duplicate_spans: Vec<Vec<usize>> = vec![];
    let mut output_text = String::new();
    for cur_paragraph in 0..num_paragraphs_plus1-1 {
        let to_remove = if total_ngram_counts[cur_paragraph] == 0 {
            false
        } else {
            contained_ngram_counts[cur_paragraph] as f64 / total_ngram_counts[cur_paragraph] as f64 >= filtering_threshold
        };
        if to_remove {
            // If we should remove, keep track of duplicate spans
            duplicate_spans.push(vec![newlines[cur_paragraph], newlines[cur_paragraph + 1]]);
        } else {
            // If we should NOT remove, add to new text, and insert hsahes into bloom filter
            output_text.push_str(&text[newlines[cur_paragraph]..newlines[cur_paragraph+1]]);
            for hash in &total_ngram_hashes[cur_paragraph] {
                if !no_update_bloom_filter {
                    bloom_filter.insert_hashes(&hash);
                }
            }
        }

        // And then handle the extra things to delete:
        while overflow_idxs.len() > 0 {

            let cur_overflow = overflow_idxs[overflow_idxs.len() - 1];
            let cur_hash = &overflow_hashes[overflow_hashes.len() - 1];
            // If fully preceding current then it has survived thusfar and should be added in and we should be done processing
            if cur_overflow.1 < newlines[cur_paragraph] {
                if !no_update_bloom_filter {
                    bloom_filter.insert_hashes(cur_hash);
                }
                overflow_idxs.pop();
                overflow_hashes.pop();
            } else { // intersects
                if to_remove {
                    overflow_idxs.pop();
                    overflow_hashes.pop();
                } else {
                    break;
                }
            }
        }
    }

    // Anything left in the overflow survived to the end and should kept? 
    while overflow_hashes.len() > 0 {
        let cur_hash = overflow_hashes.pop().unwrap();
        if !no_update_bloom_filter {
            bloom_filter.insert_hashes(&cur_hash);
        }
    }



    // And then finally make the output 
    if annotate {
        data["bff_duplicate_spans"] = serde_json::to_value(duplicate_spans).unwrap();
        data["bff_contained_ngram_count"] = total_contains.into();
    } else {
        data["text"] = Value::String(output_text.trim_end().to_string());
    }   

    (data, total_bytes - output_text.len() as usize, total_bytes as usize)

}




fn process_line_oldboth(line: &String, bloom_filter: &BloomFilter, min_ngram_size: usize, max_ngram_size:usize, 
                        filtering_threshold: f64, no_update_bloom_filter: bool, annotate: bool) 
    -> (serde_json::Value, usize, usize){
    let mut data: Value = serde_json::from_str(&line).unwrap();
    let mut total_items = 0;
    let mut removed_items = 0;
    let text = data["text"].as_str().unwrap();


    let newlines = if false {
        vec![0, text.len()]
    } else {
        let mut newlines = Vec::new();
        newlines.push(0);
        for i in text.match_indices('\n') {
            newlines.push(i.0);
        }
        newlines.push(text.len());
        newlines
    };
    let mut windows_to_remove = Vec::new();

    let mut total_ngrams = 0;
    let mut total_contained_ngrams = 0;
    for paragraph_window in newlines.windows(2) {
        let paragraph = &text[paragraph_window[0]..paragraph_window[1]];
        total_items += 1;

        // calculate hashes for the paragraph
        let mut hashes: Vec<Vec<u64>> = Vec::new();
        let mut ngram: VecDeque<&str> = VecDeque::with_capacity(max_ngram_size);
        for token in tokenize(paragraph) {
            ngram.push_back(token);
            // If not hashing whole paragraphs, add ngrams to the bloom filter as they reach max size
            if ngram.len() >= max_ngram_size {
                hashes.push(bloom_filter.hashes(&ngram));
                ngram.pop_front();
            }
        }

        // If the paragraph was too short, put in a shorter ngram, so we can dedupe short
        // paragraphs exactly.
        if hashes.is_empty() && ngram.len() >= min_ngram_size {
            hashes.push(bloom_filter.hashes(&ngram));
        }

        let contained_ngrams = hashes
            .iter()
            .filter(|ngram| bloom_filter.contains_hashes(ngram))
            .count();
        total_ngrams += hashes.len();
        total_contained_ngrams += contained_ngrams;

        // calculate how many ngrams are in the bloom filter
        let number_of_ngrams = hashes.len();

        // produce output
        let too_many_duplicate_ngrams =
            contained_ngrams as f64 / number_of_ngrams as f64 > filtering_threshold;
        if too_many_duplicate_ngrams {
            windows_to_remove.push(paragraph_window);
            removed_items += 1;
        } else if !no_update_bloom_filter {
            for ngram in hashes {
                bloom_filter.insert_hashes(&ngram);
            }
        }
    }

    // if annotate_attribute_only or annotate_only, add the annotation to the json
    if annotate {
        data["bff_duplicate_spans"] = serde_json::to_value(windows_to_remove).unwrap();
        data["bff_contained_ngram_count"] =
            serde_json::to_value(total_contained_ngrams).unwrap();
    } else {
        let mut output_paragraphs = String::new();
        let mut last_end = 0;
        for paragraph_window in windows_to_remove {
            output_paragraphs.push_str(&text[last_end..paragraph_window[0]]);
            last_end = paragraph_window[1];
        }
        output_paragraphs.push_str(&text[last_end..]);
        if (total_contained_ngrams as f64) / (total_ngrams as f64) > filtering_threshold {
            output_paragraphs = String::new(); // If we found enough duplicates to remove whole document too
        }
        data["text"] = Value::String(output_paragraphs);
        data["bff_contained_ngram_count_before_dedupe"] =
            serde_json::to_value(total_contained_ngrams).unwrap();
    }

    if annotate {
        // Allowed fields
        let allowed_fields = [
            "bff_duplicate_spans",
            "bff_contained_ngram_count",
            "id",
            "source",
            "text"
        ];

        // Iterate through the keys of the JSON object and remove any field that is not in the allowed_fields list
        if let Value::Object(ref mut map) = data {
            map.retain(|key, _| allowed_fields.contains(&key.as_str()));
            }
    }
    (data, removed_items, total_items)
}




fn process_line_exact(line: &String, bloom_filter: &BloomFilter, no_update_bloom_filter: bool, annotate: bool) ->
    (serde_json::Value, usize, usize) {
    // Hacky "exact dedup" using bloom filters
    // Just hashes the WHOLE text 

    let mut data: Value = serde_json::from_str(&line).unwrap();
    let mut removed_bytes = 0;

    let text = data["text"].as_str().unwrap();
    let total_bytes = text.len();
    let mut fake_ngram: VecDeque<&str> = VecDeque::with_capacity(1);
    fake_ngram.push_back(text);
    let hashes = bloom_filter.hashes(&fake_ngram);

    if bloom_filter.contains_hashes(&hashes) {
        // If we've seen this before, modify text XOR annotate 
        if annotate {
            data["bff_exact_duplicate"] = serde_json::to_value(true).unwrap();
        } else {
            data["text"] = Value::String(String::new());
        }
        removed_bytes = total_bytes;
    } else {
        // If we haven't seen this before, insert it
        if !no_update_bloom_filter {
            bloom_filter.insert_hashes(&hashes);
        }
    }   

    return (data, removed_bytes, total_bytes)
}




/*========================================================
=                       I/O Stuff                        =
========================================================*/

async fn expand_dirs(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {    
    // Can handle both local and s3 files/directories
    // Note that this function is async because we need to wait for all the s3 files to get expanded
    // Also note that the output vector is SORTED (in case S3 has inconsistent list_objects_v2 ordering)
    let mut files: Vec<PathBuf> = Vec::new();
    let suffices = vec![".gz", "", ".zstd", ".zst"];
    for path in paths {
        if is_s3(path) {
            let s3_result = expand_s3_dirs(path).await?;
            for file in s3_result {
                files.push(file.clone());
            }
        } else if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
            for suffix in &suffices {
                for entry in glob(&format!("{}/**/*.jsonl{}", path_str, suffix))? {
                    files.push(entry?.to_path_buf());
                }
            }

        } else {
            files.push(path.clone());
        }
    }   
    files.sort();
    Ok(files)
}



async fn expand_s3_dirs(s3_uri: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut s3_files: Vec<PathBuf> = Vec::new();

    let (bucket, prefix) = split_s3_path(s3_uri);
    let region_provider = RegionProviderChain::default_provider();
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;
    let client = Client::new(&config);

    let mut response = client
        .list_objects_v2()    
        .bucket(bucket.to_owned())
        .prefix(prefix.to_owned())
        .into_paginator()
        .send();

    while let Some(result) = response.next().await {
        match result {
            Ok(output) => {
                for object in output.contents() {
                    let key = object.key().unwrap();
                    if !(key.ends_with(".jsonl.gz") || key.ends_with(".jsonl") || key.ends_with(".jsonl.zstd") || key.ends_with(".jsonl.zst")) {
                        continue;
                    }
                    let mut s3_file = PathBuf::from("s3://");
                    s3_file.push(bucket.clone());
                    s3_file.push(key);
                    s3_files.push(s3_file);
                }
            }
            Err(err) => {
                eprintln!("Error collecting S3 files | {err:?}")
            }
        }
    }
    Ok(s3_files)
}


async fn get_object_with_retry(bucket: &str, key: &str, num_retries: usize) -> Result<GetObjectOutput, aws_sdk_s3::Error> {
    let mut attempts = 0;
    let base_delay = Duration::from_millis(100);
    let max_delay = Duration::from_millis(2000);

    let mut rng = rand::thread_rng();
    let client = get_s3_client().await;
    loop {
        match client.get_object().bucket(bucket).key(key).send().await {
            Ok(response) => return Ok(response),
            Err(e) if attempts < num_retries => {
                // Calculate delay for exponential backoff, add some randomness so multiple threads don't access at the
                // same time.
                println!("Error {}/{}: {}", e, attempts, num_retries);
                let random_delay =  rng.gen_range(Duration::from_millis(0)..Duration::from_millis(1000));
                let mut exponential_delay = base_delay * 2u32.pow(attempts as u32);
                if exponential_delay > max_delay {
                    exponential_delay = max_delay;
                }
                sleep(exponential_delay + random_delay).await;
                attempts += 1;
            }
            Err(e) => {
                println!("Too many errors reading: {}. Giving up.", key);
                return Err(e.into());
            }
        }
    }
}



async fn get_reader_from_s3(path: &PathBuf, num_retries: Option<usize>) -> Result<BufReader<Cursor<Vec<u8>>>>{
    // Gets all the data from an S3 file and loads it into memory and returns a Bufreader over it
    let num_retries = num_retries.unwrap_or(5);
    let (s3_bucket, s3_key) = split_s3_path(path);
    let object = get_object_with_retry(&s3_bucket, &s3_key, num_retries).await?;
    let body_stream = object.body.into_async_read();
    let mut data = Vec::new();

    if (path.extension().unwrap() == "zstd") || (path.extension().unwrap() == "zst") {
        let zstd = asyncZstd::new(body_stream);
        let mut reader = tBufReader::with_capacity(1024 * 1024, zstd);
        reader.read_to_end(&mut data).await.expect("Failed to read data {:path}");
    } else if path.extension().unwrap() == "gz" {
        let gz = asyncGZ::new(body_stream);
        let mut reader = tBufReader::with_capacity(1024 * 1024, gz);
        reader.read_to_end(&mut data).await.expect("Failed to read data {:path}");
    } else {
        let mut reader = tBufReader::with_capacity(1024 * 1024, body_stream);
        reader.read_to_end(&mut data).await.expect("Failed to read data {:path}");
    }

    let cursor = Cursor::new(data);
    Ok(BufReader::new(cursor))
}

fn create_dir_if_not_exists(path: &PathBuf) -> Result<(), std::io::Error> {
    if is_s3(path) {
        return Ok(()); // S3 bucket/prefixes always already exist
    }
    match create_dir_all(path) {
        Ok(_) => Ok(()),
        Err(err) => {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                Ok(())
            } else {
                Err(err)
            }
        }
    }
}

fn split_s3_path(path: &PathBuf) -> (String, String) {
    // Splits s3_uri into (bucket, key)
    let path_str = path.to_str().expect("Invalid path");

    let path_without_scheme = path_str
        .strip_prefix("s3://")
        .expect("Path must start with 's3://'");

    let slash_index = path_without_scheme
        .find('/')
        .expect("Path must contain a slash after the bucket name");

    let bucket = &path_without_scheme[..slash_index];
    let key = &path_without_scheme[slash_index + 1..];
    (bucket.to_string(), key.to_string())
}


fn is_s3(path: &PathBuf) -> bool {
    if let Some(s) = path.to_str() {
        s.starts_with("s3://")
    } else {
        false
    }
}

async fn get_s3_client() -> Client {
    let region_provider = RegionProviderChain::default_provider();
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;
    Client::new(&config)
}


fn get_output_filename(inputs: &[PathBuf], input_filename: &PathBuf, output_directory: &PathBuf) -> PathBuf {
    // More fancy output-file naming that no longer assumes unique inputs
    let matching_prefix = inputs
        .iter()
        .find(|pfx| input_filename.starts_with(pfx))
        .expect("No matching prefix found?!?");

    let relative_path = input_filename.strip_prefix(matching_prefix).unwrap();
    output_directory.clone().join(relative_path)

}


fn compress_data(data: Vec<u8>, filename: &PathBuf) -> Vec<u8> {
    // Given a filename with an extension, compresses a bytestream accordingly 
    // {zst, zstd} -> zstandard, {gz} -> gzip, anything else -> nothing


    let output_data = match filename.extension().unwrap().to_str() {
        Some("gz") => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&data).unwrap();            
            encoder.finish().unwrap()
        },
        Some("zstd") | Some("zst") => {
            let mut encoder = ZstdEncoder::new(Vec::new(), 0).unwrap();
            encoder.write_all(&data).unwrap();            
            encoder.finish().unwrap()
        },
        _ => {data}
    };
    output_data
}

/*=============================================================
=                       Main Function                         =
=============================================================*/


#[tokio::main]
async fn main() -> Result<()> {
    let args = ArgParser::parse();

    match &args.command {
        Commands::Bff {inputs, output_directory, bloom_filter_file, expected_ngram_count,
                       fp_rate, min_ngram_size, max_ngram_size, substr_seqlen, filtering_threshold, 
                       remove_type, num_hashers, no_update_bloom_filter, annotate,
                       threads, no_save_bloom_filter, no_progress_bar, shard_num, total_shards} =>
        {
            assert!(shard_num < total_shards, "Shard num must be < total shards");
            bff(inputs, output_directory, bloom_filter_file, expected_ngram_count,
                fp_rate, min_ngram_size, max_ngram_size, substr_seqlen, filtering_threshold,
                remove_type, num_hashers, no_update_bloom_filter, annotate,
                threads, no_save_bloom_filter, no_progress_bar, shard_num, total_shards).await?;
        },
        Commands::Sysreq {expected_ngram_count, num_hashers, fp_rate} => {
            let bff_size = compute_bloom_size(*fp_rate, *expected_ngram_count, false, *num_hashers);
            let num_hashers = if *num_hashers == 0 {
                    BloomFilter::optimal_number_of_hashers(bff_size, *expected_ngram_count)
                } else {*num_hashers};
            println!("To handle {} tokens with fp rate {}, you'd need a filter of size {} and {} hashers",
                     expected_ngram_count, fp_rate, human_bytes(bff_size as f64), num_hashers);            
        },

    }
    Ok(())
}



async fn bff(inputs: &Vec<PathBuf>, output_directory: &PathBuf, bloom_filter_file: &Option<PathBuf>,
       expected_ngram_count: &usize, fp_rate: &f64, min_ngram_size: &usize, max_ngram_size: &usize,
       substr_seqlen: &usize, filtering_threshold: &f64, remove_type: &RemoveType, num_hashers: &usize, 
       no_update_bloom_filter: &bool, annotate: &bool, threads: &usize, no_save_bloom_filter: &bool,
       no_progress_bar: &bool, shard_num: &usize, total_shards: &usize) -> Result<()> {


    // SETUP PHASE:
    // Set up {output_location, filter, inputs, threading, progress bar}
    let start_time = Instant::now();
    create_dir_if_not_exists(output_directory).unwrap(); 
    let bloom_filter = Arc::new(BloomFilter::from_args(bloom_filter_file.clone(), *expected_ngram_count, 
                                *fp_rate, *num_hashers)); 


    // Setup input files
    let all_inputs = expand_dirs(inputs).await?;

    let mut shard: Vec<PathBuf> = Vec::new();
    let mut idx = *shard_num;
    while idx < all_inputs.len() {
        shard.push(all_inputs[idx].clone());
        idx += total_shards;
    }
    let mut rng = thread_rng();
    shard.shuffle(&mut rng); 
    // Setup threads
    let threads = if *threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        *threads
    };
    // Setup progress bar
    let pbar = ProgressBar::new(shard.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    if !no_progress_bar {
        pbar.lock().unwrap().inc(0); // Makes pbar show up with 0/N files complete
    }    
    println!("Completed setup phase in {:?} seconds", start_time.elapsed().as_secs());


    // LOOP PHASE(using threadpool)
    let loop_start_time = Instant::now();
    let total_bytes = Arc::new(Mutex::new(0));
    let removed_bytes = Arc::new(Mutex::new(0));
    let threadpool = ThreadPool::new(threads);
    for input in shard {
        //let output = output_directory.clone().join(input.file_name().unwrap());
        let output = get_output_filename(inputs, &input, output_directory);
        let bloom_filter = bloom_filter.clone();
        let pbar_option: Option<Arc<Mutex<ProgressBar>>> = if *no_progress_bar {
            None
        } else {
            Some(pbar.clone())
        };
        let min_ngram_size = min_ngram_size.clone();
        let max_ngram_size = max_ngram_size.clone();
        let substr_seqlen = substr_seqlen.clone();
        let filtering_threshold = filtering_threshold.clone();
        let remove_type = remove_type.clone();
        let no_update_bloom_filter = no_update_bloom_filter.clone();
        let annotate = annotate.clone();
        let no_progress_bar = no_progress_bar.clone();
        let total_bytes = Arc::clone(&total_bytes);
        let removed_bytes = Arc::clone(&removed_bytes);


        threadpool.execute(move || {
            if no_progress_bar {
                println!("Processing {input:?}...");
            }

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();            
            let result = rt.block_on(
                process_file(
                    &input,
                    &output,
                    &bloom_filter,
                    max_ngram_size,
                    min_ngram_size,
                    substr_seqlen, 
                    &remove_type,
                    filtering_threshold.clone(),
                    no_update_bloom_filter.clone(),
                    annotate.clone(),
                    &pbar_option
                    )                
                );
            match result {
                Ok(outputs) => {
                    let (removed_doc_bytes, total_doc_bytes) = outputs;
                    let mut total_guard = total_bytes.lock().unwrap();
                    *total_guard += total_doc_bytes;
                    let mut removed_guard = removed_bytes.lock().unwrap();
                    *removed_guard += removed_doc_bytes;                        
                },
                Err(err) => {
                    eprintln!("Error processing {:?}; {:?}", input, err);
                }
            }        
        });
    }
    threadpool.join();
    println!("Completed filtering all files in {:?} seconds", 
             loop_start_time.elapsed().as_secs());    


    // FINALIZE PHASE 
    // Save bloom filter
    match &bloom_filter_file {
        Some(path) => {
            if !no_update_bloom_filter && !no_save_bloom_filter {
                let write_start_time = Instant::now();
                println!("Writing bloom filter to {:?}...", path);
                bloom_filter.write_to_file(&path).unwrap();
                println!("...Bloom filter written in {:?} seconds.", write_start_time.elapsed().as_secs());                
            }

        }
        _ => {}
    }
    // Print out summary
    println!("After running, BFF sparsity was {:?}", bloom_filter.calculate_sparsity());
    println!("Completed full BFF run in {:?} seconds", start_time.elapsed().as_secs());

    let total_bytes = *total_bytes.lock().unwrap();
    let removed_bytes = *removed_bytes.lock().unwrap();
    println!("Stats: Saw {} of text | Removed {} of them",
             human_bytes(total_bytes as f64), removed_bytes as f64 / total_bytes as f64);   
    Ok(())
}