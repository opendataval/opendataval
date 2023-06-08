from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from opendataval.dataloader.register import Register, _from_pandas, cache

CHALLENGE_URL = "https://opendataval.yongchanstat.com/challenge"
"""Backend URL for opendataval to get signed URLs to the challenge data set."""


def gen_paths(
    signed_urls: list[dict[str, str]], cache_dir: str, force_download: bool
) -> Iterator[Path]:
    for entry in signed_urls:
        file_name, signed_url = entry["name"], entry["signed_url"]
        yield cache(signed_url, cache_dir, file_name, force_download)


@Register("challenge-iris", cacheable=True, one_hot=True)
def iris_challenge(cache_dir: str, force_download: bool):
    resp = requests.post(f"{CHALLENGE_URL}/challengeiris").json()
    # Iterator must be unpacked
    (train_path,) = gen_paths(resp["table"], cache_dir, force_download)
    df = pd.read_csv(train_path)
    df["species"] = df["species"].astype("category").cat.codes
    return _from_pandas(df, "species")
