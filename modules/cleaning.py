import os
import pandas as pd
import numpy as np
import re

from dotenv import load_dotenv

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK once inside container
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def clean_dataset(
    raw_csv_path="/opt/airflow/data/raw/Amazon_Reviews.csv",
    save_csv_path="/opt/airflow/data/clean/cleaned_reviews.csv",
):
    """
    Clean dataset using EXACT steps from notebook.
    """

    # Ensure output dir exists
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

    # Load dataset exactly like notebook
    df = pd.read_csv(
        raw_csv_path,
        engine="python",
        on_bad_lines="skip"
    )

    # ---------- 1. Country mapping ----------
    country_map = {
        'US': 'United States',
        'GB': 'United Kingdom',
        'AU': 'Australia',
        'JP': 'Japan',
        'CA': 'Canada',
        'ZA': 'South Africa',
        'IN': 'India',
        'BE': 'Belgium',
        'AE': 'United Arab Emirates',
        'NZ': 'New Zealand',
        'IL': 'Israel',
        'NL': 'Netherlands',
        'SE': 'Sweden',
        'DK': 'Denmark',
        'JM': 'Jamaica',
        'DE': 'Germany',
        'MY': 'Malaysia',
        'FR': 'France',
        'ES': 'Spain',
        'IT': 'Italy',
        'PR': 'Puerto Rico',
        'KW': 'Kuwait',
        'PK': 'Pakistan',
        'PT': 'Portugal',
        'TW': 'Taiwan',
        'IE': 'Ireland',
        'DO': 'Dominican Republic',
        'TR': 'Turkey',
        'HK': 'Hong Kong',
        'ME': 'Montenegro',
        'FI': 'Finland',
        'CH': 'Switzerland',
        'CO': 'Colombia',
        'CY': 'Cyprus',
        'PA': 'Panama',
        'ID': 'Indonesia',
        'EG': 'Egypt',
        'HR': 'Croatia',
        'BR': 'Brazil',
        'GR': 'Greece',
        'PL': 'Poland',
        'NI': 'Nicaragua',
        'TH': 'Thailand',
        'SK': 'Slovakia',
        'CN': 'China',
        'VG': 'British Virgin Islands',
        'BH': 'Bahrain',
        'CZ': 'Czech Republic',
        'GG': 'Guernsey',
        'AT': 'Austria',
        'MA': 'Morocco',
        'CV': 'Cape Verde',
        'GE': 'Georgia',
        'LT': 'Lithuania',
        'MT': 'Malta',
        'MC': 'Monaco',
        'RU': 'Russia',
        'UA': 'Ukraine',
        'AG': 'Antigua and Barbuda',
        'OM': 'Oman',
        'VI': 'U.S. Virgin Islands',
        'VN': 'Vietnam',
        'KH': 'Cambodia',
        'NG': 'Nigeria',
        'QA': 'Qatar',
        'NO': 'Norway',
        'GH': 'Ghana',
        'CR': 'Costa Rica',
        'BO': 'Bolivia',
        'MQ': 'Martinique',
        'KE': 'Kenya',
        'PE': 'Peru',
        'BS': 'Bahamas',
        'RO': 'Romania',
        'JE': 'Jersey',
        'PH': 'Philippines',
        'DZ': 'Algeria',
        'RS': 'Serbia',
        'AZ': 'Azerbaijan',
        'AF': 'Afghanistan',
        'SA': 'Saudi Arabia',
        'AM': 'Armenia',
        'MV': 'Maldives',
        'EE': 'Estonia',
        'HU': 'Hungary',
        'CM': 'Cameroon',
        'PG': 'Papua New Guinea',
        'AR': 'Argentina',
        'BG': 'Bulgaria',
        'MX': 'Mexico',
        'LV': 'Latvia',
        'SL': 'Sierra Leone',
        'CD': 'Congo (DRC)',
        'MO': 'Macau',
        'GI': 'Gibraltar',
        'LU': 'Luxembourg',
        'AO': 'Angola',
        'BD': 'Bangladesh',
        'KR': 'South Korea',
        'KZ': 'Kazakhstan',
        'ET': 'Ethiopia',
        'FJ': 'Fiji',
        'SG': 'Singapore',
        'UG': 'Uganda',
        'SR': 'Suriname',
        'EC': 'Ecuador',
        'BY': 'Belarus',
        'SI': 'Slovenia',
        'TN': 'Tunisia',
        'HN': 'Honduras',
        'VE': 'Venezuela',
        'NP': 'Nepal',
        'BB': 'Barbados',
        'CW': 'Curacao',
        'GF': 'French Guiana',
        'BZ': 'Belize',
        'CL': 'Chile',
        'GT': 'Guatemala',
        'KG': 'Kyrgyzstan',
        'MG': 'Madagascar',
        'IM': 'Isle of Man',
        'TT': 'Trinidad and Tobago',
        'PY': 'Paraguay',
        'LK': 'Sri Lanka',
        'BA': 'Bosnia and Herzegovina',
        'LA': 'Laos',
        'TZ': 'Tanzania',
        'IQ': 'Iraq',
        'BQ': 'Bonaire, Saint Eustatius and Saba',
        'GY': 'Guyana',
        'MN': 'Mongolia',
        'IS': 'Iceland',
        'MD': 'Moldova',
        'UY': 'Uruguay',
        'SO': 'Somalia',
        'RW': 'Rwanda',
        'MU': 'Mauritius',
        'BM': 'Bermuda',
        'LB': 'Lebanon',
        'IR': 'Iran',
        'JO': 'Jordan',
        'SV': 'El Salvador',
        'BW': 'Botswana',
        'AD': 'Andorra',
        'CI': "Côte d'Ivoire",
        'ZM': 'Zambia',
        'MK': 'North Macedonia',
        'MM': 'Myanmar'
    }

    def convert_country(code):
        if pd.isna(code) or code is None or str(code).strip().lower() in ["", "nan", "none"]:
            return "Do not mention"
        code = str(code).strip().upper()
        return country_map.get(code, "Do not mention")

    df["Country"] = df["Country"].apply(convert_country)

    # ---------- 2. Drop Profile Link ----------
    if "Profile Link" in df.columns:
        df = df.drop(columns=["Profile Link"])

    # ---------- 3. Drop missing review title/text ----------
    df = df.dropna(subset=["Review Title", "Review Text"])

    # ---------- 4. Fill Date of Experience ----------
    if "Date of Experience" in df.columns and "Review Date" in df.columns:
        df["Date of Experience"] = df["Date of Experience"].fillna(df["Review Date"])

    # ---------- 5. Combine Title + Text ----------
    df["full_review"] = df["Review Title"].astype(str) + " " + df["Review Text"].astype(str)

    # ---------- 6. Lowercase ----------
    df["full_review"] = df["full_review"].apply(lambda x: x.lower())
    df['full_review'] = df['full_review'].str.replace('\W', ' ', regex=True)

    # ---------- 7. Remove numbers ----------
    df["full_review"] = df["full_review"].apply(lambda x: re.sub(r"\d+", "", x))

    # ---------- 8. Remove stopwords ----------
    sw = stopwords.words("english")
    df["full_review"] = df["full_review"].apply(
        lambda x: " ".join(word for word in x.split() if word not in sw)
    )

    # ---------- 9. Remove extra spaces ----------
    df["full_review"] = df["full_review"].str.replace(r"\s+", " ", regex=True).str.strip()

    # ---------- 10. Lemmatization ----------
    lemmatizer = WordNetLemmatizer()

    def lemmatize(sentence):
        token = sentence.split()
        return " ".join(lemmatizer.lemmatize(word) for word in token)

    df["full_review"] = df["full_review"].apply(lemmatize)

    # ---------- 11. Save cleaned data ----------
    df.to_csv(save_csv_path, index=False)

    print(f"[CLEAN] Saved cleaned dataset to: {save_csv_path}")
    return save_csv_path
