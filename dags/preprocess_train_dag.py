from airflow.decorators import dag, task
from datetime import datetime
import os

# Import modules
from modules.preprocess import preprocess_dataset
from modules.train import train_models
from modules.db_io import save_split_to_db


# ==== CONFIG ====
CLEANED_CSV = "/opt/airflow/data/clean/cleaned_reviews.csv"
SPLIT_DIR = "/opt/airflow/data/splits"
VECTORIZER_DIR = "/opt/airflow/data/vectorizers"
MODEL_DIR = "/opt/airflow/data/models"


@dag(
    dag_id="preprocess_train_dag",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["amazon", "ml", "training"],
)
def preprocess_train_pipeline():

    @task()
    def preprocess_task():
        """
        - train/test split
        - tfidf word + tfidf ngram
        - save vectorizers
        - save csv splits
        """
        result_paths = preprocess_dataset(
            cleaned_csv_path=CLEANED_CSV,
            split_dir=SPLIT_DIR,
            vectorizer_dir=VECTORIZER_DIR,
        )
        return result_paths

    @task()
    def save_splits_to_db(_):
        """
        Upload split data (X_train, X_test, y_train, y_test)
        to PostgreSQL cloud db.
        """
        save_split_to_db(split_dir=SPLIT_DIR)
        return "Train/test splits saved to DB."

    @task()
    def train_task(_):
        """
        Train 2 models:
        - Logistic Regression word-level
        - Logistic Regression ngram-level
        Save as .pkl
        """
        result = train_models(
            split_dir=SPLIT_DIR,
            vectorizer_dir=VECTORIZER_DIR,
            model_dir=MODEL_DIR,
        )
        return result

    # ==== DAG FLOW ====
    split_paths = preprocess_task()
    save_db = save_splits_to_db(split_paths)
    train_models_output = train_task(save_db)


preprocess_train_pipeline()
