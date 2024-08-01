use ahash::RandomState;
use anyhow::{anyhow, Result, Error};
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use clap::{Args, Parser, Subcommand};
use flate2::read::{MultiGzDecoder};
use flate2::write::GzEncoder;
use flate2::Compression;
use glob::glob;
use human_bytes::human_bytes;
use indicatif::{ProgressBar,ProgressStyle};
use rand::{Rng,thread_rng};
use rand::seq::SliceRandom;
use serde_json::Value;
use std::clone::Clone;
use std::collections::VecDeque;
use std::fs::{OpenOptions, remove_file, create_dir_all};
use std::hash::{BuildHasher, Hash, Hasher};
use std::io;
use std::io::{Cursor};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::mem::size_of;
use std::path::{PathBuf, Path};
use std::string::String;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant};
use std::thread::available_parallelism;
use sysinfo::{
    System,
};
use threadpool::ThreadPool;
use unicode_segmentation::UnicodeSegmentation;
use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_s3::{Client};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::operation::get_object::GetObjectOutput;
use tokio::io::{AsyncBufReadExt};
use tokio::io::BufReader as tBufReader;
use tokio::time::{Duration, sleep};
use async_compression::tokio::bufread::GzipDecoder as asyncGZ;
use rayon::prelude::*;


/*=======================================================
=                    Argument Struct                    =
=======================================================*/

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,
}


#[derive(Debug, Clone, Args)]
struct BffArgs{
    /*
      -- BLOOM FILTER KWARGS 
        + bloom_filter_location: where we save/load the bloom filter
        + expected_ngram_count: how many ngrams we're expecting
        + fp_rate: false positive (per ngram) we're expecting
      -- BLOOM FILTER HYPERPARAMS
        + min_ngram_size (default: 5), smallest ngram size to consider 
        + max_ngram_size (default: 13), largest ngram size to consider 
        + filtering_threshold (default 0.80), threshold used to determine if text is duplicate
    
      -- BLOOM FILTER OVERRIDE KWARGS: 
        + bloom_filter_size (default: 0), if >0 we force the filter to have this size
        + no_update_bloom_filter (default: false), if true, we never update the bloom filter
        + no_save_bloom_filter (default: false), if true, we don't save the bloom filter at the end
        + annotate_only (default: false), if true we leave text intact but annotate with which spans are duplicates
        + whole_document (default: false), if true, we dedup across the whole document (spanning pargaraphs)
        + whole_paragraph (default: false), if true, we don't match ngrams but rather whole paragraphs
        + no_progress (default: false), if true, we don't display a progress bar, instead printing out files as we handle them
        + threads: (default: 0), if > 0, we force use of this many threads, o/w it's automatically computed    
    */

    // Bloom filter kwargs
    #[arg(long)]
    bloom_filter_file: Option<PathBuf>,

    #[arg(required = true, long)]
    expected_ngram_count: usize,        

    #[arg(required = true, long)]
    fp_rate: f64,

    // Bloom filter hyperparams
    #[arg(long, default_value_t = 5)]
    min_ngram_size: usize,        

    #[arg(long, default_value_t = 13)]
    max_ngram_size: usize,        

    #[arg(long, default_value_t = 0.80)]
    filtering_threshold: f64,

    // Bloom filter override args
    #[arg(long, default_value_t=0)]
    bloom_filter_size: usize,        

    #[arg(long, default_value_t = false)]
    no_update_bloom_filter: bool,        

    #[arg(long, default_value_t = false)]
    no_save_bloom_filter: bool,        

    #[arg(long, default_value_t = false)]
    annotate_attribute_only: bool,            

    #[arg(long, default_value_t = RemoveType::Paragraph, value_enum)]
    remove_type: RemoveType,

    #[arg(long, default_value_t = false)]
    whole_document: bool,        

    #[arg(long, default_value_t = false)]
    whole_paragraphs: bool,

    #[arg(long, default_value_t = false)]
    no_progress: bool,

    #[arg(long, short = 't', default_value_t = 0)]
    threads: usize,    


    #[arg(long, default_value_t=0)]
    shard_num: usize,

    #[arg(long, default_value_t=1)]
    total_shards: usize,
}



#[derive(Subcommand, Debug)]
enum Commands {
    /* Two commands here: 
      - `bff` is for LOCAL files (local in -> local out)
      - `bff_remote` is for S3 files (S3 in -> S3 out)
    Where each takes default arguments of: 


    And then subcommand arguments
    -- bff:
        + inputs: file or files (directories okay) of gzip compressed newline-delimited JSON files with a 'text' field
        + output_directory: where the deduplicated files get loaded to
    
    -- bff_remote:
        + bucket
        + input_dir
        + output_dir
    */

    #[clap(arg_required_else_help = true)]
    Bff {
        // subcommand arguments
        #[arg(required=true, long)]
        inputs: Vec<PathBuf>,

        #[arg(required=true, long)]
        output_directory: PathBuf,    

        #[command(flatten)]
        bff_args: BffArgs, 
    },

    BffRemote {
        #[arg(required=true, long)]
        bucket: String,

        #[arg(required=true, long)]
        input_dir: String,

        #[arg(required=true, long)]
        output_dir: String,

        #[command(flatten)]
        bff_args: BffArgs,

        // Local number of retries; we try to load each file from s3 this many times.
        #[arg(long, default_value_t=3)]
        num_retries: usize,

        // Global number of retries; we do a full loop through all remaining files this many times.
        // i.e., 
        // remaining = all_paths
        // for i in num_retries:
        //     remaining = process_data(remaining) 
        #[arg(long, default_value_t=3)]
        num_global_retries: usize,
    },

    Sysreq {
        #[arg(required=true, long)]
        expected_ngram_count: usize,
        #[arg(required=true, long)]
        fp_rate: f64
    },

}


#[derive(Debug, Clone, Eq, PartialEq, clap::ValueEnum)]
enum RemoveType {
    // Types for what we check to see if is a duplicate
    Paragraph, // Paragraph level only
    Document,  // Whole document only
    Both,      // Does paragraph first, but if enough of the ngrams are contained in the bff, removes the whole document
               // NOTE: ^ will add some ngram data (OF TO-REMOVE ngrams) into the filter [other methods don't do this]
}



/*===================================================
=                     Bloom Filter stuff            =
===================================================*/

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


    fn from_args(bff_args: &BffArgs) -> Self {
        /* Uses a BFFArgs object to build a bloom filter 
        Logic:
            - Check if file exists, if so, just load it and return 
            - Get size:
                + if size is explicitly speciifed, use this
                + otherwise, compute based on ngrams + fp rate 
            - Return 
        */        
        let mut bloom_filter_size = bff_args.bloom_filter_size;

        let bloom_filter = match &bff_args.bloom_filter_file {
            Some(path) if path.exists() => {
                println!("Loading bloom filter from {:?}...", path);
                BloomFilter::from_file(&path).unwrap()
            }
            _ => {
            println!("Creating new bloom filter...");
            if bff_args.bloom_filter_size == 0 {
                bloom_filter_size = compute_bloom_size(bff_args.fp_rate, bff_args.expected_ngram_count, true);
            }
            let num_hashers = BloomFilter::optimal_number_of_hashers(
                bloom_filter_size,
                bff_args.expected_ngram_count,
            );
            BloomFilter::new(bloom_filter_size, num_hashers)
            }
        };

 

        println!("Bloom filter has size {} | FP Rate {:?}",
                 human_bytes(bloom_filter.size_in_bytes() as f64), 
                 bloom_filter.my_prob_of_false_positive(bff_args.expected_ngram_count));
        bloom_filter
    }
}




fn compute_bloom_size(fp_rate: f64, expected_ngram_count: usize, limit_to_sys: bool) -> usize {
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


    // Save some time by checking endpoint first
    if limit_to_sys && BloomFilter::prob_of_false_positive(hi, expected_ngram_count, 
                                           BloomFilter::optimal_number_of_hashers(hi, expected_ngram_count)) > fp_rate {
        println!(
            "WARNING: To achieve desired false-positive rate, you'd need >90% of system RAM. Defaulting to 90% \
            system RAM.");
        return hi;
    }

    // Then do binary search to find optimal size
    while lo < hi-1 {
        let mid = lo + (hi - lo) / 2;
        let num_hashers = BloomFilter::optimal_number_of_hashers(mid, expected_ngram_count);
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


#[allow(clippy::too_many_arguments)]
fn process_file(
    input_file: &PathBuf,
    output_file_path: &PathBuf,
    bloom_filter: &Arc<BloomFilter>,
    bff_args: &BffArgs,
    pbar_option: &Option<Arc<Mutex<ProgressBar>>>,
) -> Result<(usize, usize), io::Error> {

    // Setup input/output writers
    let input_file = OpenOptions::new()
        .read(true)
        .write(false)
        .create(false)
        .open(input_file)?;
    let reader = BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(input_file));


    let output_file = OpenOptions::new()
        .read(false)
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_file_path)?;
    let mut writer = BufWriter::with_capacity(
        1024 * 1024,
        GzEncoder::new(output_file, Compression::default()),
    );


    // Loop over lines and do BFF stuff
    let mut count = 0;
    let mut fully_skipped = 0;
    let mut removed_items = 0;
    let mut total_items = 0;
    for line in reader.lines() {
        count += 1;
        let (dedup_data, removed_line_items, total_line_items) = process_line(&line.unwrap(), &bloom_filter, &bff_args);
        removed_items += removed_line_items;
        total_items += total_line_items;

        if dedup_data.get("text").unwrap().as_str().unwrap().trim().is_empty() {
            fully_skipped += 1;
        }
        else {
            serde_json::to_writer(&mut writer, &dedup_data)?;
            writer.write_all(b"\n")?;             
        }        
    }

    if count == fully_skipped {
        remove_file(output_file_path)?;
    }


    match pbar_option {
        Some(pbar) => {
            let pb = pbar.lock().unwrap();
            pb.inc(1);
            if pb.position() < 10 || pb.position() % 100 == 0 {
                println!("Log Progress: {}/{} - {} elapsed, ETA {}",
                    pb.position(), pb.length().unwrap(),
                    format_duration(pb.elapsed()),
                    format_duration(pb.eta()));
            }
        }
        None => (),
    }
    Ok((removed_items, total_items))
}

fn format_duration(dur: Duration) -> String {
    format!("{:02}:{:02}:{:02}", dur.as_secs() / 3600, (dur.as_secs() % 3600) / 60, dur.as_secs() % 60)
}

async fn get_object_with_retry(client: &Client, bucket: &str, key: &str, num_retries: usize) -> Result<GetObjectOutput, aws_sdk_s3::Error> {
    let mut attempts = 0;
    let base_delay = Duration::from_millis(100);
    let max_delay = Duration::from_millis(2000);

    let mut rng = rand::thread_rng();

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

async fn process_file_s3(
    s3_bucket: &String,
    s3_input: &String,
    s3_output: &String,
    bloom_filter: &Arc<BloomFilter>,
    bff_args: &BffArgs,
    pbar_option: &Option<Arc<Mutex<ProgressBar>>>,
    num_retries: usize,
) -> Result<(usize, usize), Error> {
    // Phase 1a: Build s3 client
    let region_provider = RegionProviderChain::default_provider();
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;
    let client = Client::new(&config);

    let object = get_object_with_retry(&client, s3_bucket, s3_input, num_retries).await?;
    let body_stream = object.body.into_async_read();
    let gz = asyncGZ::new(body_stream);
    let reader = tBufReader::with_capacity(1024 * 1024, gz);
    let mut lines_iter = reader.lines();
    let mut all_lines = Vec::new();
    while let Some(line) = lines_iter.next_line().await? {
        all_lines.push(line);
    }

    // Phase 1c: Setup output buffer to upload->s3 eventually...
    // TODO: Make output writer streaming too?
    let mut output_data = Vec::new();
    let encoder = GzEncoder::new(Cursor::new(&mut output_data), Compression::default());
    let mut buf_writer = BufWriter::with_capacity(1024 * 1024, encoder);
    buf_writer.anosetuhueo();

    // Phase 2: Loop over lines, process each line, and write it if not fully eradicated
    let mut count = 0;
    let mut fully_skipped = 0;
    let mut removed_items = 0;
    let mut total_items = 0;
    for line in all_lines {
        count += 1;
        let (dedup_data, removed_line_items, total_line_items) = process_line(&line.to_string(), &bloom_filter, &bff_args);
        removed_items += removed_line_items;
        total_items += total_line_items;
        if dedup_data.get("text").unwrap().as_str().unwrap().is_empty() {
            fully_skipped += 1;
        }
        else {
            serde_json::to_writer(&mut buf_writer, &dedup_data)?;
            buf_writer.write_all(b"\n")?;          
        }
        
    }
    // println!("Number of lines in {:?} is {}", s3_input, count);


    // Phase 3: to finalize, write to s3 if there's something to write
    buf_writer.flush()?;
    let encoder = buf_writer.into_inner().expect("Failed to get encoder");
    encoder.finish().unwrap();

    if fully_skipped < count {
        let bytes_to_upload = ByteStream::from(output_data);
        client
            .put_object()
            .bucket(s3_bucket)
            .key(s3_output)
            .body(bytes_to_upload)
            .send()
            .await?;
    }
    match pbar_option {
        Some(pbar) => {
            let pb = pbar.lock().unwrap();
            pb.inc(1);
            if pb.position() < 10 || pb.position() % 100 == 0 {
                println!("Log Progress: {}/{} - {} elapsed, ETA {}",
                    pb.position(), pb.length().unwrap(),
                    format_duration(pb.elapsed()),
                    format_duration(pb.eta()));
            }
        }
        None => (),
    }
    Ok((removed_items, total_items))
}



fn process_line(line: &String, bloom_filter: &BloomFilter, bff_args: &BffArgs) -> (serde_json::Value, usize, usize){
    let mut data: Value = serde_json::from_str(&line).unwrap();
    let mut total_items = 0;
    let mut removed_items = 0;
    let text = data["text"].as_str().unwrap();


    let newlines = if bff_args.remove_type == RemoveType::Document {
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
        let mut ngram: VecDeque<&str> = VecDeque::with_capacity(bff_args.max_ngram_size);
        for token in tokenize(paragraph) {
            ngram.push_back(token);
            // If not hashing whole paragraphs, add ngrams to the bloom filter as they reach max size
            if !bff_args.whole_paragraphs && ngram.len() >= bff_args.max_ngram_size {
                hashes.push(bloom_filter.hashes(&ngram));
                ngram.pop_front();
            }
        }

        // If the paragraph was too short, put in a shorter ngram, so we can dedupe short
        // paragraphs exactly.
        if hashes.is_empty() && ngram.len() >= bff_args.min_ngram_size {
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
            contained_ngrams as f64 / number_of_ngrams as f64 > bff_args.filtering_threshold;
        if too_many_duplicate_ngrams {
            windows_to_remove.push(paragraph_window);
            removed_items += 1;
        } else if !bff_args.no_update_bloom_filter {
            for ngram in hashes {
                bloom_filter.insert_hashes(&ngram);
            }
        }
    }

    // if annotate_attribute_only or annotate_only, add the annotation to the json
    if bff_args.annotate_attribute_only {
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
        if bff_args.remove_type == RemoveType::Both &&
          (total_contained_ngrams as f64) / (total_ngrams as f64) > bff_args.filtering_threshold
        {
            output_paragraphs = String::new(); // If we found enough duplicates to remove whole document too
        }
        data["text"] = Value::String(output_paragraphs);
        data["bff_contained_ngram_count_before_dedupe"] =
            serde_json::to_value(total_contained_ngrams).unwrap();
    }

    if bff_args.annotate_attribute_only {
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




fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split_word_bounds().filter(|w| {
        for c in w.chars() {
            if !c.is_whitespace() {
                return true;
            }
        }
        false
    })
}




/*========================================================
=                       I/O Stuff                        =
========================================================*/

fn expand_dirs(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = vec![];
    for path in paths {
        if path.is_dir() {
            let path_str = path
                .to_str()
                .ok_or_else(|| anyhow!("invalid path '{}'", path.to_string_lossy()))?;
            for entry in glob(&format!("{}/**/*.json*.gz", path_str))? {
                files.push(entry?.to_path_buf());
            }
        } else {
            files.push(path.clone());
        }
    }
    Ok(files)
}

fn create_dir_if_not_exists(path: &PathBuf) -> Result<(), std::io::Error> {
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

fn extract_s3_basename(input_path: &str) -> &str {
    let parts: Vec<&str> = input_path.split('/').collect();
    parts.last().unwrap()
}



async fn gather_s3_io(bucket: &str, prefix: &str, output_dir: &str,
                      shard_num: usize, total_shards: usize) -> Result<Vec<(String, String)>, Error> {
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

    let mut io_pairs: Vec<(String, String)> = Vec::new();
    while let Some(result) = response.next().await {
        match result {
            Ok(output) => {
                for object in output.contents() {
                    let input_key = object.key().unwrap();
                    if !(input_key.ends_with(".jsonl.gz") || input_key.ends_with(".json.gz")) {
                        continue;
                    }
                    let basename = extract_s3_basename(&input_key);
                    let output_key = Path::new(output_dir).join(basename).as_os_str().to_str().unwrap().to_string();
                    let io_pair: (String, String) = (String::from(input_key), String::from(&output_key));
                    io_pairs.push(io_pair);                 
                }
            }
            Err(err) => {
                eprintln!("{err:?}")
            }
        }
    }
    // select shard before we shuffle 
    let mut shard: Vec<(String, String)> = Vec::new();    
    let mut idx = shard_num;
    while idx < io_pairs.len() {
        shard.push(io_pairs[idx].clone());
        idx += total_shards;
    }

    // Then shuffle
    let mut rng = thread_rng();
    shard.shuffle(&mut rng);

    Ok(shard)
}


/*=============================================================
=                       Main Function                         =
=============================================================*/
#[tokio::main]
async fn main() -> std::io::Result<()> {
    let args = ArgParser::parse();

    match &args.command {
        Commands::Bff {inputs, output_directory, bff_args} =>
        { 
            assert!(bff_args.shard_num < bff_args.total_shards, "Shard num must be <= total shards");
            bff(inputs, output_directory, &bff_args)?;
        },

        Commands::BffRemote {bucket, input_dir, output_dir, bff_args, num_retries, num_global_retries} => {
            assert!(bff_args.shard_num < bff_args.total_shards, "Shard num must be <= total shards");
            bff_remote(bucket, input_dir, output_dir, &bff_args, num_retries, num_global_retries).await?;
        }
        Commands::Sysreq {expected_ngram_count, fp_rate} => {
            let bff_size = compute_bloom_size(*fp_rate, *expected_ngram_count, false);
            let num_hashers = BloomFilter::optimal_number_of_hashers(bff_size, *expected_ngram_count);
            println!("To handle {} tokens with fp rate {}, you'd need a filter of size {} and {} hashers",
                     expected_ngram_count, fp_rate, human_bytes(bff_size as f64), num_hashers);
        },
    }

    Ok(())
}


fn bff(inputs: &Vec<PathBuf>, output_directory: &PathBuf, bff_args: &BffArgs) -> std::io::Result<()> {
    /*
    General pseudocode:
    Setup:
        - Build/setup the bloom filter
        - Expand all the inputs
        - Setup progress bar
    Main loop:
        - loop over all files and process them 
    Finalize:
        - Write bff if needed
    */
    // SETUP PHASE
    let start_time = Instant::now();
    create_dir_if_not_exists(output_directory).unwrap();
    let bloom_filter = Arc::new(BloomFilter::from_args(bff_args));
    let all_inputs = expand_dirs(inputs).unwrap();

    // Select shard and then shuffle
    let mut shard: Vec<PathBuf> = Vec::new();
    let mut idx = bff_args.shard_num;
    while idx < all_inputs.len() {
        shard.push(all_inputs[idx].clone());
        idx += bff_args.total_shards;
    }
    // Then shuffle
    let mut rng = thread_rng();
    shard.shuffle(&mut rng);




    let pbar = ProgressBar::new(shard.len() as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    if !bff_args.no_progress {
        pbar.lock().unwrap().inc(0); // initializes pbar
    }    
    println!("Completed setup phase in {:?} seconds", start_time.elapsed().as_secs());




    // LOOP PHASE (W/ Threadpool)
    let threads = if bff_args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        bff_args.threads
    };    
    let loop_start_time = Instant::now();
    let total_items = Arc::new(Mutex::new(0));
    let removed_items = Arc::new(Mutex::new(0));
    let threadpool = ThreadPool::new(threads);
    for input in shard {
        //let mut output = output_directory.clone();
        let output = output_directory.clone().join(input.file_name().unwrap());
        //output.push(input.file_name().unwrap());
        let bloom_filter = bloom_filter.clone();
        let bff_args = bff_args.clone();
        let total_items = Arc::clone(&total_items);
        let removed_items = Arc::clone(&removed_items);
        let pbar_option: Option<Arc<Mutex<ProgressBar>>> = if bff_args.no_progress {
            None
        } else {
            Some(pbar.clone())
        };

        threadpool.execute(move || {
            if bff_args.no_progress {
                println!("Processing {input:?}...");
            }
            let (removed_doc_items, total_doc_items) = process_file(
                &input,
                &output,
                &bloom_filter,
                &bff_args,
                &pbar_option,
            )
            .unwrap();

            let mut total_guard = total_items.lock().unwrap();
            *total_guard += total_doc_items;

            let mut removed_guard = removed_items.lock().unwrap();
            *removed_guard += removed_doc_items;

        });
    }
    threadpool.join();    
    println!("Completed filtering all files in {:?} seconds", 
             loop_start_time.elapsed().as_secs());
    

    // FINALIZE PHASE 
    match &bff_args.bloom_filter_file {
        Some(path) => {
            if (!bff_args.no_update_bloom_filter) && (!bff_args.no_save_bloom_filter) {
                let write_start_time = Instant::now();
                println!("Writing bloom filter to {:?}...", path);
                bloom_filter.write_to_file(&path).unwrap();
                println!("...Bloom filter written in {:?} seconds.", write_start_time.elapsed().as_secs());
            }
        }
        _ => {}
    }


    println!("After running, BFF sparsity was {:?}", bloom_filter.calculate_sparsity());

    println!("Completed full BFF run in {:?} seconds", start_time.elapsed().as_secs());

    let total_items = *total_items.lock().unwrap();
    let removed_items = *removed_items.lock().unwrap();
    println!("Stats: Saw {} items | Removed {} of them",
             total_items, removed_items as f64 / total_items as f64);   
    Ok(())
}



async fn bff_remote(bucket: &String, input_dir: &String, output_dir: &String, bff_args: &BffArgs, num_retries: &usize, num_global_retries: &usize) -> std::io::Result<()> {
    /*
    General pseudocode:
    Setup:
        - Build/setup the bloom filter
        - Setup thing to read s3_io
        - Setup progress bar
    Main loop:
        - loop over all files and process them 
    Finalize:
        - Write bff if needed
    */
    let start_time = Instant::now();
    let bloom_filter = Arc::new(BloomFilter::from_args(bff_args));

    let mut io_pairs = gather_s3_io(bucket, input_dir, output_dir,
                                    bff_args.shard_num, bff_args.total_shards).await.unwrap();
    println!("Collected {} input files...", io_pairs.len());

    let num_files = io_pairs.len();
    let err_count = Arc::new(Mutex::new(0));
    let pbar = ProgressBar::new(num_files as u64)
        .with_style(
            ProgressStyle::with_template(
                "Files {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]",
            ).unwrap()
        );
    let pbar = Arc::new(Mutex::new(pbar));
    println!("Completed setup phase in {:?} seconds", start_time.elapsed().as_secs());

    if !bff_args.no_progress {
        pbar.lock().unwrap().inc(0); // initializes pbar
    }


    let loop_start_time = Instant::now();
    let total_items = Arc::new(Mutex::new(0));
    let removed_items = Arc::new(Mutex::new(0));    
    let threads = if bff_args.threads == 0 {
        available_parallelism().unwrap().get()
    } else {
        bff_args.threads
    };    
    let threadpool = ThreadPool::new(threads);

    for retry_count in 0..*num_global_retries {
        let failed_io_pairs: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let mut rng = rand::thread_rng();
        for io_pair in &io_pairs {
            let num_retries = (*num_retries).clone();
            let num_global_retries = (*num_global_retries).clone();
            let retry_count = retry_count.clone();
            let bucket = bucket.clone();
            let bloom_filter = bloom_filter.clone();
            let bff_args = bff_args.clone();
            let failed_io_pairs = Arc::clone(&failed_io_pairs);
            let err_count: Arc<Mutex<i32>> = Arc::clone(&err_count);
            let total_items = Arc::clone(&total_items);
            let removed_items = Arc::clone(&removed_items);        
            let pbar_option: Option<Arc<Mutex<ProgressBar>>> = if bff_args.no_progress {
                None
            } else {
                Some(pbar.clone())
            };

            let (input_path, output_path) = io_pair.clone();
            threadpool.execute(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                let result = rt.block_on(
                            process_file_s3(&bucket, 
                                &input_path, 
                                &output_path,
                                &bloom_filter,
                                &bff_args, 
                                &pbar_option,
                                num_retries)
                            );
                match result {
                    Ok(outputs) => {
                        let (rem_doc_items, tot_doc_items) = outputs;
                        let mut total_guard = total_items.lock().unwrap();
                        *total_guard += tot_doc_items;
                        let mut removed_guard = removed_items.lock().unwrap();
                        *removed_guard += rem_doc_items;
                    }
                    Err(err) => {
                            eprintln!("Round {}/{}: Error processing {}; {:?}", retry_count+1, num_global_retries, input_path, err);
                            if retry_count < num_global_retries - 1 {
                                // in all but last round, push the failed pair to failed_io_pairs
                                let mut fail_guard = failed_io_pairs.lock().unwrap();
                                fail_guard.push((input_path, output_path));
                            } else {
                                // in last round, give up and mark this one as an error
                                let mut count = err_count.lock().unwrap();
                                *count += 1;                                
                            }

                        }                
                    }
            });          
            // Wait a little before spawning the next processor.
            let random_delay = rng.gen_range(Duration::from_millis(0)..Duration::from_millis(100));
            sleep(random_delay).await;
        }
        threadpool.join();
        io_pairs = failed_io_pairs.lock().unwrap().clone();  
    }
    println!("Completed filtering all files in {:?} seconds", 
             loop_start_time.elapsed().as_secs());

    // FINALIZE PHASE 
    match &bff_args.bloom_filter_file {
        Some(path) => {
            if (!bff_args.no_update_bloom_filter) && (!bff_args.no_save_bloom_filter) {
                let write_start_time = Instant::now();
                println!("Writing bloom filter to {:?}...", path);
                bloom_filter.write_to_file(&path).unwrap();
                println!("...Bloom filter written in {:?} seconds.", write_start_time.elapsed().as_secs());
            }
        }
        _ => {}
    }
    
    println!("Error count is {}/{}", err_count.lock().unwrap(), num_files);
    println!("After running, BFF sparsity was {:?}", bloom_filter.calculate_sparsity());

    println!("Completed full BFF run in {:?} seconds", start_time.elapsed().as_secs());
    let total_items = *total_items.lock().unwrap();
    let removed_items = *removed_items.lock().unwrap();
    println!("Stats: Saw {} items | Removed {} of them",
             total_items, removed_items as f64 / total_items as f64);    
    Ok(())
}


