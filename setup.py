from __future__ import annotations
import os
import urllib.request
import tarfile
import shutil
from setuptools.command.install import install
from setuptools import setup, find_packages
from retrie.retrie import Blacklist
import pickle
import re
import nltk
import boto3


PROJECT_ROOT = os.path.dirname(__file__)

class BaselineInstall(install):
    def run(self):
        # Perform the standard installation
        install.run(self)
        # Custom setup for baseline
        print("Setting up baseline requirements...")
        nltk.download('punkt')

class TrainingInstall(install):
    def run(self):
        install.run(self)
        # Download and prepare training data
        print("Downloading training data...")
        nltk.download('punkt')

class EvalInstall(install):
    def run(self):
        install.run(self)
        print("Preparing evaluation setup...")

class DevInstall(install):
    def run(self):
        install.run(self)
        print("Setting up development environment...")

class DownloadAssetsCommand(install):
    description = 'download and set up larger assets (e.g., models, banlists) after installation'

    user_options = install.user_options + [
        ('skip-downloads=', 's', "whether to skip all downloads"),
        ('skip-model-downloads=', None, "whether to skip model downloads"),
        ('skip-banlist-downloads=', None, "whether to skip banlist downloads"),
        ('rw-banlist-type=', None, "whether to use the curated banlist or the uncurated banlist")
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.skip_downloads = None
        self.skip_model_downloads = None
        self.skip_banlist_downloads = None
        self.rw_banlist_type = 'curated'

    def finalize_options(self):
        install.finalize_options(self)

        assert self.skip_downloads in [None, 'y', 'yes', '1', 't', 'true']
        assert self.skip_model_downloads in [None, 'y', 'yes', '1', 't', 'true']
        assert self.skip_banlist_downloads in [None, 'y', 'yes', '1', 't', 'true']
        assert self.rw_banlist_type in ['curated', 'uncurated']

        if self.skip_downloads:
            self.skip_model_downloads = 'yes'
            self.skip_banlist_downloads = 'yes'


    def run(self):
        # Call the parent class to perform the installation
        super().run()

        # Download punkt which is necessary for some mappers
        nltk.download('punkt')

        if not self.skip_model_downloads:
            # Download the models
            print("\n\nReached model downloads\n\n")
            self._download_fasttext_model()
            self._download_quality_models()

        # Download the RefinedWeb banlists
        if not self.skip_banlist_downloads:
            print("\n\nReached banlist downloads\n\n")
            if self.rw_banlist_type == 'curated':
                self._download_curated_refinedweb_banlists()
            elif self.rw_banlist_type == 'uncurated':
                self._create_refinedweb_banlists()

    def _download_fasttext_model(self):
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        MODEL_SUBDIRECTORY = "baselines/mappers/enrichers/language_id_enrichment_models"
        MODEL_FILENAME = "lid.176.bin"
        destination = os.path.join(PROJECT_ROOT, MODEL_SUBDIRECTORY, MODEL_FILENAME)

        if not os.path.exists(destination):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            print(f'Downloading {url} to {destination}')
            urllib.request.urlretrieve(url, destination)
            print(f"Finsihed downloading {url} to {destination}")
        else:
            print(f'File {destination} already exists')

    def _download_quality_models(self):
        MODEL_SUBDIRECTORY = "baselines/mappers/enrichers/quality_prediction_enrichment_models"

        # Models and their URLs
        models = {
            #"model.bin": "https://wmtis.s3.eu-west-1.amazonaws.com/quality_prediction_model/model.bin",
            "fasttext_oh_eli5.bin": "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
            "en.arpa.bin": "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin",
            "en.sp.model": "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.sp.model",
        }

        for MODEL_FILENAME, url in models.items():
            destination = os.path.join(PROJECT_ROOT, MODEL_SUBDIRECTORY, MODEL_FILENAME)

            if not os.path.exists(destination):
                print(f"Downloading {MODEL_FILENAME} to {destination}...")
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                urllib.request.urlretrieve(url, destination)
                print(f"Finished downloading {MODEL_FILENAME} to {destination}")
            else:
                print(f"File {destination} already exists")

    def _download_curated_refinedweb_banlists(self):
        CURATED_BANLIST_PATH = "baselines/mappers/banlists/refinedweb_banned_domains_curated.txt"
        CURATED_BANLIST_URL = "https://huggingface.co/datasets/mlfoundations/refinedweb_banned_domains_curated/resolve/main/refinedweb_banned_domains_curated.txt"
        if not os.path.exists(CURATED_BANLIST_PATH):
            print("Downloading curated banlist necessary to run refinedweb.yaml...")
            os.makedirs(os.path.dirname(CURATED_BANLIST_PATH), exist_ok=True)
            urllib.request.urlretrieve(CURATED_BANLIST_URL, CURATED_BANLIST_PATH)
            print(f"Finished downloading {os.path.basename(CURATED_BANLIST_PATH)} to {CURATED_BANLIST_PATH}")
        else:
            print(f"Curated banlist for refinedweb already exists at {CURATED_BANLIST_PATH}")

    def _create_refinedweb_banlists(self):
        UNCURATED_BANLISTS_URL = "ftp://ftp.ut-capitole.fr/pub/reseau/cache/squidguard_contrib/blacklists.tar.gz"
        BANLIST_OUTPUT_DIR = "baselines/mappers/banlists"
        BANNED_CATEGORIES = [
            'adult',
            'phishing',
            'dating',
            'gambling',
            'filehosting',
            'ddos',
            'agressif',
            'chat',
            'mixed_adult',
            'arjel'
        ]

        if not os.path.exists(f"{BANLIST_OUTPUT_DIR}/refinedweb_banned_domains_and_urls.txt"):
            print(f"Downloading {UNCURATED_BANLISTS_URL}...")
            urllib.request.urlretrieve(UNCURATED_BANLISTS_URL, f"{BANLIST_OUTPUT_DIR}/blacklists.tar.gz")

            print("Extracting banlists...")
            with tarfile.open(f"{BANLIST_OUTPUT_DIR}/blacklists.tar.gz") as file:
                file.extractall(f"{BANLIST_OUTPUT_DIR}")

            print("Building banlist from target categories...")
            banned_domains = []
            banned_urls = []
            for category in BANNED_CATEGORIES:
                if os.path.exists(f"{BANLIST_OUTPUT_DIR}/blacklists/{category}/domains"):
                    with open(f"{BANLIST_OUTPUT_DIR}/blacklists/{category}/domains", "r") as file:
                            banned_domains.extend(file.read().splitlines())

                if os.path.exists(f"{BANLIST_OUTPUT_DIR}/blacklists/{category}/urls"):
                    with open(f"{BANLIST_OUTPUT_DIR}/blacklists/{category}/urls", "r") as file:
                            banned_urls.extend(file.read().splitlines())
            banlist = banned_domains + banned_urls

            # Removes the raw downloads (with all the different categories)
            os.remove(f"{BANLIST_OUTPUT_DIR}/blacklists.tar.gz")
            shutil.rmtree(f'{BANLIST_OUTPUT_DIR}/blacklists')


            print("Writing banlists to files...")
            with open(f"{BANLIST_OUTPUT_DIR}/refinedweb_banned_domains.txt", "w") as file:
                for item in banned_domains:
                    file.write(f"{item}\n")

            with open(f"{BANLIST_OUTPUT_DIR}/refinedweb_banned_urls.txt", "w") as file:
                for item in banned_urls:
                    file.write(f"{item}\n")

            with open(f"{BANLIST_OUTPUT_DIR}/refinedweb_banned_domains_and_urls.txt", "w") as file:
                for item in banlist:
                    file.write(f"{item}\n")

            banlist = [b.lower() for b in banlist]
            pattern = re.compile(Blacklist(banlist, match_substrings=True).compiled)
            with open(f"{BANLIST_OUTPUT_DIR}/refinedweb_banned_domains_and_urls_regex.pkl", "wb") as file:
                pickle.dump(pattern, file)

        else:
            print(f"File {f'{BANLIST_OUTPUT_DIR}/refinedweb_banned_domains_and_urls.txt'} already exists")


with open('requirements.txt') as f:
    required = [r for r in f.read().splitlines() if 'github' not in r]

setup(
    name='baselines',  # Change this to your package name
    version='0.0.1',  # Change this to your package version
    description='Description of your package',  # Add a brief description
    packages=find_packages(),
    install_requires=required,
    cmdclass={
        'install': DownloadAssetsCommand,
    },
)

setup(
    name='your_package',
    version='0.1.0',
    description='Your package description',
    packages=find_packages(),
    install_requires=required,
    extras_require={
        'baselines': ['some_package>=1.0.0'],
        'training': ['tensorflow>=2.0', 'keras'],
        'eval': ['scikit-learn'],
        'dev': ['pytest', 'sphinx']
    },
    cmdclass={
        'install': DownloadAssetsCommand,
        'install_baselines': BaselineInstall,
        'install_training': TrainingInstall,
        'install_eval': EvalInstall,
        'install_dev': DevInstall
    },
)
