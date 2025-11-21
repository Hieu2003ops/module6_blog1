import os
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv


def extract_kaggle_dataset(
    dataset: str,
    download_dir: str = "/opt/airflow/data"
):
    """
    Download a Kaggle dataset and return the path to extracted files.
    
    Parameters
    ----------
    dataset : str
        Kaggle dataset identifier. Example: "dongrelaxman/amazon-reviews-dataset".
    download_dir : str
        Directory where dataset will be downloaded inside the Airflow container.
    
    Returns
    -------
    str
        Path to the downloaded dataset folder.
    """

    # Load Kaggle credentials from .env
    load_dotenv()
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise ValueError("Missing Kaggle credentials. Ensure KAGGLE_USERNAME and KAGGLE_KEY exist in .env")

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Ensure target folder exists
    os.makedirs(download_dir, exist_ok=True)

    # Download dataset (zip file) and unzip=True to extract
    api.dataset_download_files(
        dataset,
        path=download_dir,
        unzip=True
    )

    print(f"Dataset downloaded to: {download_dir}")
    return download_dir
