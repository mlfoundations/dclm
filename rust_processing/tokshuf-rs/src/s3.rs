// General tools for interfacing with s3
use std::path::{PathBuf, Path};
use anyhow::{Result};

use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_s3::{Client, Error as S3Error};
use aws_sdk_s3::operation::put_object::PutObjectOutput;
use aws_sdk_s3::primitives::ByteStream;
use async_compression::tokio::bufread::GzipDecoder as asyncGZ;
use async_compression::tokio::bufread::ZstdDecoder as asyncZstd;
use std::io::{BufReader, Cursor};
use rand::{Rng};
use tokio::io::AsyncReadExt;
use tokio::io::BufReader as tBufReader;
use tokio::time::{Duration, sleep};


/*==========================================================
=            String/PathBuf synchronous methods            =
==========================================================*/


pub(crate) fn is_s3<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().to_str().map_or(false, |s| s.starts_with("s3://"))
}


pub(crate) fn split_s3_path<P: AsRef<Path>>(path: P) -> (String, String) {
    // Splits s3_uri into (bucket, key)
    let path_str = path.as_ref().to_str().expect("Invalid path");

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


/*============================================================
=              Asynchronous S3 Methods                       =
============================================================*/


async fn s3_retry<T, F, Fut>(max_retries: usize, mut operation: F) -> Result<T, S3Error> 
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, S3Error>>,
{
    let mut rng = rand::thread_rng();
    let base_delay = Duration::from_millis(100);
    let max_delay = Duration::from_millis(2000);    
    let mut attempts = 0;
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) if attempts < max_retries => {
                println!("Error {}/{}: {}", err, attempts, max_retries);
                let random_delay =  rng.gen_range(Duration::from_millis(0)..Duration::from_millis(1000));
                let mut exponential_delay = base_delay * 2u32.pow(attempts as u32);
                if exponential_delay > max_delay {
                    exponential_delay = max_delay;
                }
                sleep(exponential_delay + random_delay).await;
                attempts += 1;
            }, 
            Err(err) => return Err(err.into())
        }
    }
}


pub(crate) async fn get_s3_client() -> Result<Client, S3Error> {
    // Gets a client from default configs (setup with awscli)
    let region_provider = RegionProviderChain::default_provider();
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;
    Ok(Client::new(&config))
}


pub(crate) async fn expand_s3_dir(s3_uri: &PathBuf, valid_exts: &[&str]) -> Result<Vec<PathBuf>, S3Error> {
    // Collects all .json.gz/.jsonl.gz files prefixed by the provided s3_uri 
    let mut s3_files: Vec<PathBuf> = Vec::new();
    let client = get_s3_client().await?;
    let (bucket, prefix) = split_s3_path(s3_uri);

    //let ext = ext.unwrap_or(".jsonl.gz");
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
                    let key = object.key().unwrap_or_default();     
                    if valid_exts.iter().any(|ext| key.ends_with(ext)) {
                        let mut s3_file = PathBuf::from("s3://");
                        s3_file.push(bucket.clone());
                        s3_file.push(key);
                        s3_files.push(s3_file);
                    }
                }
            }
            Err(err) => {
                eprintln!("Error collecting S3 files | {err:?}");
                return Err(err.into());
            }
        }
    }
    Ok(s3_files)
}


pub(crate) async fn count_s3_dirsize(s3_uri: &PathBuf) -> Result<usize, S3Error> {
    let client = get_s3_client().await?;
    let (bucket, prefix) = split_s3_path(s3_uri);
    let mut response = client
        .list_objects_v2()    
        .bucket(bucket.to_owned())
        .prefix(prefix.to_owned())
        .into_paginator()
        .send();

    let mut total_size = 0;
    while let Some(result) = response.next().await {
        match result {
            Ok(output) => {
                for object in output.contents() {
                    total_size += object.size().unwrap_or_default();
                }
            }
            Err(err) => {
                eprintln!("Error calculating s3 dir size | {err:?}");
                return Err(err.into());
            }
        }
    }
    Ok(total_size as usize)
}


async fn get_object_with_retry(bucket: &str, key: &str, num_retries: usize) -> Result<ByteStream, S3Error> {
    let client = get_s3_client().await?;
    s3_retry(num_retries, || async {
        let output = client.get_object().bucket(bucket).key(key).send().await?;
        Ok(output.body)        
    })
    .await
}



pub(crate) async fn get_reader_from_s3<P: AsRef<Path>>(path: P, num_retries: Option<usize>) -> Result<BufReader<Cursor<Vec<u8>>>, S3Error>{
    // Gets all the data from an S3 file and loads it into memory and returns a Bufreader over it
    let (s3_bucket, s3_key) = split_s3_path(&path);
    let object_body = get_object_with_retry(&s3_bucket, &s3_key, num_retries.unwrap_or(5)).await?;
    let body_stream = object_body.into_async_read();
    let mut data = Vec::new();

    if (path.as_ref().extension().unwrap() == "zstd") || (path.as_ref().extension().unwrap() == "zst") {
        let zstd = asyncZstd::new(body_stream);
        let mut reader = tBufReader::with_capacity(1024 * 1024, zstd);
        reader.read_to_end(&mut data).await.expect("Failed to read data {:path}");

    } else if path.as_ref().extension().unwrap() == "gz" {
        let gz = asyncGZ::new(body_stream);
        let mut reader = tBufReader::with_capacity(1024 * 1024, gz);
        reader.read_to_end(&mut data).await.expect("Failed to read data {:path}");        
    } else {
        let mut reader = tBufReader::with_capacity(1024 * 1024, body_stream);
        reader.read_to_end(&mut data).await.expect("Failed to read data {:path}");
    };

    let cursor = Cursor::new(data);

    Ok(BufReader::new(cursor))
}



pub(crate) async fn write_cursor_to_s3(s3_uri: &PathBuf, cursor: Cursor<Vec<u8>>) -> Result<PutObjectOutput, S3Error> {
    let (s3_bucket, s3_key) = split_s3_path(s3_uri);
    let client = get_s3_client().await?;
    let bytes = ByteStream::from(cursor.into_inner());
    let response = client
            .put_object()
            .bucket(s3_bucket)
            .key(s3_key)
            .body(bytes)
            .send()
            .await?;   

    Ok(response)
}


