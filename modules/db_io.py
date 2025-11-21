import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


def get_postgres_engine():
    """
    Create a SQLAlchemy engine using environment variables.
    Works with Railway, Neon, Supabase, or local Postgres.
    """
    load_dotenv()

    host = os.getenv("POSTGRES_HOST")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    port = os.getenv("POSTGRES_PORT", "5432")

    if not all([host, user, password, db]):
        raise ValueError("Missing PostgreSQL environment variables.")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(url)

    return engine


def save_cleaned_data_to_db(cleaned_csv_path, table_name="cleaned_reviews"):
    """
    Save the cleaned CSV into PostgreSQL.
    """
    engine = get_postgres_engine()

    df = pd.read_csv(cleaned_csv_path)

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )

    print(f"[DB_IO] Saved cleaned data to table: {table_name}")


def save_split_to_db(split_dir="/opt/airflow/data/split"):
    """
    Save X_train, X_test, y_train, y_test into PostgreSQL.
    Each is saved as a separate DB table.
    """

    engine = get_postgres_engine()

    mapping = {
        "X_train.csv": "split_x_train",
        "X_test.csv": "split_x_test",
        "y_train.csv": "split_y_train",
        "y_test.csv": "split_y_test",
    }

    for file_name, table in mapping.items():
        file_path = os.path.join(split_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {file_path}")

        df = pd.read_csv(file_path)

        df.to_sql(
            table,
            engine,
            if_exists="replace",
            index=False
        )

        print(f"[DB_IO] Saved {file_name} → table {table}")
