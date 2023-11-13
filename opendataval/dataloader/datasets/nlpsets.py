"""NLP data sets.

Uses HuggingFace
`transformers <https://huggingface.co/docs/transformers/index>`_. as dependency.
"""
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from opendataval.dataloader.register import Register, cache
from opendataval.dataloader.util import FolderDataset, ListDataset
from opendataval.util import batched


def BertEmbeddings(
    func: Callable[[str, bool], tuple[Sequence[str], np.ndarray]], batch_size: int = 128
):
    """Convert text data into pooled embeddings with DistilBERT model.

    Given a data set with a list of string, such as NLP data set function (see below),
    converts the sentences into strings. It is the equivalent of training a downstream
    task with bert but all the BERT layers are frozen. It is advised to just
    train with the raw strings with a BERT model located in models/bert.py or defining
    your own model. DistilBERT is just a faster version of BERT

    References
    ----------
    .. [1] J. Devlin, M.W. Chang, K. Lee, and K. Toutanova,
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        arXiv.org, 2018. Available: https://arxiv.org/abs/1810.04805.
    .. [2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf,
        DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
        arXiv.org, 2019. Available: https://arxiv.org/abs/1910.01108.
    """

    def wrapper(
        cache_dir: str, force_download: bool, *args, **kwargs
    ) -> tuple[torch.Tensor, np.ndarray]:
        from transformers import DistilBertModel, DistilBertTokenizerFast

        BERT_PRETRAINED_NAME = "distilbert-base-uncased"  # TODO update this

        cache_dir = Path(cache_dir)
        embed_path = cache_dir / f"{func.__name__}_embed"

        dataset, labels = func(cache_dir, force_download, *args, **kwargs)

        if FolderDataset.exists(embed_path):
            return FolderDataset.load(embed_path), labels

        # Slow down on gpu vs cpu is quite substantial, uses gpu accel if available
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PRETRAINED_NAME)
        bert_model = DistilBertModel.from_pretrained(BERT_PRETRAINED_NAME).to(device)
        folder_dataset = FolderDataset(embed_path)

        for batch_num, batch in enumerate(tqdm(batched(dataset, n=batch_size))):
            bert_inputs = tokenizer.__call__(
                batch,
                max_length=200,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            bert_inputs = {inp: bert_inputs[inp] for inp in tokenizer.model_input_names}

            with torch.no_grad():
                pool_embed = bert_model(**bert_inputs)[0]
                word_embeddings = pool_embed.detach().cpu()[:, 0]

            folder_dataset.write(batch_num, word_embeddings)

        folder_dataset.save()
        return folder_dataset, np.array(labels)

    return wrapper


@Register("bbc", cacheable=True, one_hot=True)
def download_bbc(cache_dir: str, force_download: bool):
    """Classification data set registered as ``"bbc"``.

    Predicts type of article from the article. Used in NLP data valuation tasks.

    References
    ----------
    .. [1] D. Greene and P. Cunningham,
        Practical Solutions to the Problem of Diagonal Dominance in
        Kernel Document Clustering, Proc. ICML 2006.
    """
    github_url = (
        "https://raw.githubusercontent.com/"
        "mdsohaib/BBC-News-Classification/master/bbc-text.csv"
    )
    filepath = cache(github_url, cache_dir, "bbc-text.csv", force_download)
    df = pd.read_csv(filepath)

    label_dict = {
        "business": 0,
        "entertainment": 1,
        "sport": 2,
        "tech": 3,
        "politics": 4,
    }
    labels = np.fromiter((label_dict[label] for label in df["category"]), dtype=int)

    return ListDataset(df["text"].values), labels


@Register("imdb", cacheable=True, one_hot=True)
def download_imdb(cache_dir: str, force_download: bool):
    """Binary category sentiment analysis data set registered as ``"imdb"``.

    Predicts sentiment analysis of the review as either positive (1) or negative (0).
    Used in NLP data valuation tasks.

    References
    ----------
    .. [1] A. Maas, R. Daly, P. Pham, D. Huang, A. Ng, and C. Potts.
        Learning Word Vectors for Sentiment Analysis.
        The 49th Annual Meeting of the Association for Computational Linguistics (2011).
    """
    github_url = (
        "https://raw.githubusercontent.com/"
        "Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    )
    filepath = cache(github_url, cache_dir, "imdb.csv", force_download)
    df = pd.read_csv(filepath)

    label_dict = {"negative": 0, "positive": 1}
    labels = np.fromiter((label_dict[label] for label in df["sentiment"]), dtype=int)

    return ListDataset(df["review"].values), labels


bbc_embedding = Register("bbc-embeddings", True, True)(BertEmbeddings(download_bbc))
"""Classification data set registered as ``"bbc-embeddings"``, BERT text embeddings."""

imdb_embedding = Register("imdb-embeddings", True, True)(BertEmbeddings(download_imdb))
"""Classification data set registered as ``"imdb-embeddings"``, BERT text embeddings."""
