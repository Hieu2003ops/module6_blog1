from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
import os

# Import modules
from modules.extract import extract_kaggle_dataset
from modules.cleaning import clean_dataset
from modules.db_io import save_cleaned_data_to_db


# ===== CONFIG =====
RAW_DATA_DIR = "/opt/airflow/data/raw"
CLEAN_DATA_DIR = "/opt/airflow/data/clean"
RAW_FILE_PATH = f"{RAW_DATA_DIR}/Amazon_Reviews.csv"
CLEAN_FILE_PATH = f"{CLEAN_DATA_DIR}/cleaned_reviews.csv"

DATASET_NAME = "dongrelaxman/amazon-reviews-dataset"


@dag(
    dag_id="extract_clean_dag",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["amazon", "etl", "cleaning"],
)
def extract_clean_pipeline():

    @task()
    def extract_task():
        """Download Kaggle dataset → RAW folder"""
        extract_kaggle_dataset(
            dataset=DATASET_NAME,
            download_dir=RAW_DATA_DIR
        )
        return RAW_FILE_PATH

    @task()
    def clean_task(raw_path: str):
        """Clean → save cleaned CSV → return path"""
        os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

        cleaned_csv_path = clean_dataset(
            raw_csv_path=raw_path,
            save_csv_path=CLEAN_FILE_PATH
        )

        return cleaned_csv_path

    @task()
    def load_clean_to_db(clean_csv_path: str):
        """Load → PostgreSQL"""
        save_cleaned_data_to_db(
            cleaned_csv_path=clean_csv_path,
            table_name="amazon_reviews_clean"
        )
        return "Saved cleaned data to DB."

    # DAG flow
    raw_file = extract_task()
    cleaned_file = clean_task(raw_file)
    load_clean_to_db(cleaned_file)


extract_clean_pipeline()
