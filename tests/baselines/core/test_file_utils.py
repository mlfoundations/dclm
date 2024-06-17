import os
import shutil

import boto3
import pytest
from cloudpathlib import S3Path
from moto import mock_s3

from baselines.core.constants import CONTENT
from baselines.core.file_utils import read_jsonl, write_jsonl, is_s3, makedirs_if_missing, delete_file, is_exists, list_dir

# You need to set up the bucket for mocked S3.
S3_TEST_BUCKET = "test-bucket"
S3_TEST_PATH = f"s3://{S3_TEST_BUCKET}/test.jsonl"


@pytest.fixture(scope="function")
def s3_setup_teardown():
    with mock_s3():
        boto3.client('s3').create_bucket(Bucket=S3_TEST_BUCKET)
        yield


@pytest.mark.parametrize("file_path", ["/tmp/test.jsonl", S3_TEST_PATH])
def test_read_write_jsonl(file_path, s3_setup_teardown):
    # Given
    data_to_write = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

    # When
    write_jsonl(data_to_write, file_path)

    # Then
    read_data = list(read_jsonl(file_path))
    assert read_data == data_to_write


def test_is_s3():
    assert is_s3("s3://bucket/path")
    assert not is_s3("/local/path/to/file")
    assert not is_s3("path/to/file")


def test_makedirs_if_missing_local():
    local_path = "/tmp/test_dir"

    # Ensure directory is not present initially
    if os.path.isdir(local_path):
        shutil.rmtree(local_path)

    makedirs_if_missing(local_path)
    assert os.path.isdir(local_path)

    # Cleanup
    shutil.rmtree(local_path)


def test_makedirs_if_missing_s3():
    s3_path = "s3://my-bucket/path/to/file.txt"
    makedirs_if_missing(s3_path)
    # For S3, we can't assert directory creation since it's a no-op.
    # We're mainly checking that no errors are raised.


# Mock the S3 environment
@pytest.fixture
def mock_s3_bucket():
    with mock_s3():
        s3 = boto3.resource('s3', region_name='us-east-1')
        s3.create_bucket(Bucket='test-bucket')
        yield


def test_delete_local_file(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello, World!")
    assert file.exists()
    delete_file(str(file))
    assert not file.exists()


@pytest.mark.usefixtures('mock_s3_bucket')
def test_delete_s3_file():
    file_path = "s3://test-bucket/test.txt"
    s3_path = S3Path(file_path)
    s3_path.write_text("Hello, S3!")
    assert s3_path.exists()
    delete_file(file_path)
    assert not s3_path.exists()


@pytest.mark.usefixtures('mock_s3_bucket')
def test_delete_nonexistent_s3_file():
    file_path = "s3://test-bucket/nonexistent.txt"
    s3_path = S3Path(file_path)
    assert not s3_path.exists()
    with pytest.raises(FileNotFoundError):
        delete_file(file_path)


def test_delete_nonexistent_local_file(tmp_path):
    file = tmp_path / "nonexistent.txt"
    assert not file.exists()
    with pytest.raises(FileNotFoundError):
        delete_file(str(file))


def test_exists_local_file(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Hello, World!")
    assert is_exists(str(file)) == True


def test_not_exists_local_file(tmp_path):
    file = tmp_path / "nonexistent.txt"
    assert is_exists(str(file)) == False


@pytest.mark.usefixtures('mock_s3_bucket')
def test_exists_s3_file():
    file_path = "s3://test-bucket/test.txt"
    s3_path = S3Path(file_path)
    s3_path.write_text("Hello, S3!")
    assert is_exists(file_path) == True


@pytest.mark.usefixtures('mock_s3_bucket')
def test_not_exists_s3_file():
    file_path = "s3://test-bucket/nonexistent.txt"
    assert is_exists(file_path) == False



def test_list_local_dir(tmp_path):
    # Create some files, including hidden ones
    (tmp_path / "file1.txt").write_text(CONTENT)
    (tmp_path / ".hidden_file").write_text("hidden content")
    listed_files = list_dir(str(tmp_path))
    assert len(listed_files) == 1
    assert "file1.txt" in listed_files[0]

@pytest.mark.usefixtures('mock_s3_bucket')
def test_list_s3_dir():
    # Create some files in S3, including hidden ones
    s3_dir = S3Path("s3://test-bucket/dir/")
    (s3_dir / "file1.txt").write_text(CONTENT)
    (s3_dir / ".hidden_file").write_text("hidden content")
    listed_files = list_dir("s3://test-bucket/dir/")
    print(listed_files)  # For debugging
    assert len(listed_files) == 1
    assert "file1.txt" in listed_files[0]
