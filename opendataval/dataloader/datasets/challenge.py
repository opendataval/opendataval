import time
from pathlib import Path

import pandas as pd
import requests

from opendataval.dataloader.register import Register, _from_pandas, cache

CHALLENGE_URL = "https://opendataval.yongchanstat.com/challenge"
"""Backend URL for opendataval to get drive ids to the challenge data set."""

RETRY_ATTEMPTS = 5


def _dataset_url(drive_id: str):
    """Get drive URL to download the challenge data set."""
    return f"https://drive.google.com/uc?id={drive_id}&?confirm=1"


def _challenge_ids(challenge: str) -> list[dict[str, str]]:
    """Get challenge ids from the opendataval backend."""
    return requests.get(f"{CHALLENGE_URL}/{challenge}").json()


def download_drive(name: str, drive_id: str, cache_dir: Path, force_download: str):
    """Downloads file from google drive with set retry attempts."""
    download_url = _dataset_url(drive_id)
    cache_dir = Path(cache_dir)

    for i in range(RETRY_ATTEMPTS):
        try:
            return cache(download_url, cache_dir, name, force_download)
        except requests.HTTPError as ex:
            print(f"Attempt {i+1} failed: {ex}, retrying in 10 seconds")
            time.sleep(10)
    else:  # Means we've exhausted all our retries
        raise TimeoutError(
            "Retry attempt exceeded, cannot download challenge dataset now, "
            "Set force_download=False and wait a few minutes to avoid rate limit."
        )


def basename(file_name: str):
    """Get basename of file."""
    return str(Path(file_name).with_suffix(""))


@Register("challenge-iris", cacheable=True, one_hot=True, presplit=True)
def iris_challenge(cache_dir: str, force_download: bool):
    drive_ids = _challenge_ids("challenge-iris")

    data = {}
    for row in drive_ids:
        filepath = download_drive(row["name"], row["id"], cache_dir, force_download)

        df = pd.read_csv(filepath)
        df["species"] = df["species"].astype("category").cat.codes
        data[basename(row["name"])] = _from_pandas(df, "species")

    return zip(data["train"], data["valid"], data["test"])
