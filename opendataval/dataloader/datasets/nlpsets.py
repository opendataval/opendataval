"""NLP data sets.

Uses HuggingFace
`transformers <https://huggingface.co/docs/transformers/index>`_. as dependency.
"""
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from opendataval.dataloader.register import Register, cache
from opendataval.dataloader.util import ListDataset

MAX_DATASET_SIZE = 10000
"""Data Valuation algorithms can take a long time for large data sets, thus cap size."""


def BertEmbeddings(func: Callable[[str, bool], tuple[ListDataset, np.ndarray]]):
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
        embed_file_name = f"{func.__name__}_{MAX_DATASET_SIZE}_embed.pt"
        embed_path = cache_dir / embed_file_name

        dataset, labels = func(cache_dir, force_download, *args, **kwargs)
        subset = np.random.RandomState(10).permutation(len(dataset))

        if embed_path.exists():
            nlp_embeddings = torch.load(embed_path)
            return nlp_embeddings, labels[subset[: len(nlp_embeddings)]]

        labels = labels[subset[:MAX_DATASET_SIZE]]
        entries = [entry for entry in dataset[subset[:MAX_DATASET_SIZE]]]

        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PRETRAINED_NAME)
        bert_model = DistilBertModel.from_pretrained(BERT_PRETRAINED_NAME)

        res = tokenizer.__call__(
            entries, max_length=200, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            pooled_embeddings = bert_model(res.input_ids, res.attention_mask)[0][:, 0]

        torch.save(pooled_embeddings.detach(), embed_path)
        return pooled_embeddings, np.array(labels)

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
