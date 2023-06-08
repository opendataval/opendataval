from pathlib import Path

import pandas as pd
import requests

from opendataval.dataloader.register import Register, _from_pandas, cache

CHALLENGE_URL = "https://opendataval.yongchanstat.com/challenge"
"""Backend URL for opendataval to get signed URLs to the challenge data set."""


def download_paths(
    signed_urls: list[dict[str, str]], cache_dir: str, force_download: bool
):
    """Downloads the the data from the cloud bucket via singed URLs."""
    for entry in signed_urls:
        file_name, signed_url = entry["name"], entry["signed_url"]
        cache(signed_url, cache_dir, file_name, force_download)


@Register("challenge-iris", cacheable=True, one_hot=True, presplit=True)
def iris_challenge(cache_dir: str, force_download: bool):
    resp = requests.post(f"{CHALLENGE_URL}/challengeiris").json()
    download_paths(resp["table"], cache_dir, force_download)

    data = []
    for type_ in ("train", "valid", "test"):
        file_path = Path(cache_dir) / "challengeiris" / f"{type_}.csv"
        df = pd.read_csv(file_path)
        df["species"] = df["species"].astype("category").cat.codes
        data.append(_from_pandas(df, "species"))
    return zip(*data)
