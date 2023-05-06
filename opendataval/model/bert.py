from collections import OrderedDict
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
)

from opendataval.dataloader import CatDataset
from opendataval.model.api import Model


class BertClassifier(Model, nn.Module):
    """Fine tune a pre-trained DistilBERT model on a classification task.

    DistilBERT is just a smaller/lighter version of BERT meant to be fine-tuned onto
    other language tasks

    References
    ----------
    .. [1] J. Devlin, M.W. Chang, K. Lee, and K. Toutanova,
        BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        arXiv.org, 2018. Available: https://arxiv.org/abs/1810.04805.
    .. [2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf,
        DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
        arXiv.org, 2019. Available: https://arxiv.org/abs/1910.01108.

    Parameters
    ----------
    pretrained_model_name : str
        Huggingface model directory containing the pretrained model for BERT
        by default "distilbert-base-uncased" [2]
    num_classes : int, optional
        Number of prediction classes, by default 2
    dropout_rate : float, optional
        Dropout rate for the embeddings of bert, helps in fine tuning, by default 0.2
    num_train_layers : int, optional
        Number of Bert layers to fine-tune. Minimum number is 1, by default 1
    """

    def __init__(
        self,
        pretrained_model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        num_train_layers: int = 2,
    ):
        super().__init__()

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)

        self.num_classes = num_classes
        self.max_length = 50  # TODO
        hidden_dim = self.bert.config.hidden_size

        # Classifier layer as specified by the HuggingFace BERT Classifiers
        classifier_dict = OrderedDict()
        classifier_dict["pre_linear"] = nn.Linear(hidden_dim, hidden_dim)
        classifier_dict["acti"] = nn.ReLU()
        classifier_dict["dropout"] = nn.Dropout(dropout_rate)
        classifier_dict["linear"] = nn.Linear(hidden_dim, num_classes)
        classifier_dict["output"] = nn.Softmax(-1)
        self.classifier = nn.Sequential(classifier_dict)

        # Freezing the embeddings and initial layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        for layer in self.bert.transformer.layer[: -max(num_train_layers, 1)]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass through DistilBert with inputs from DistilBERT tokenizer output.

        NOTE this is only applicable for a DistilBERT model that doesn't require
        ``token_type_ids``.

        Parameters
        ----------
        input_ids : torch.Tensor
            List of token ids to be fed to a model.
            [Input IDs?](https://huggingface.co/transformers/glossary#input-ids)
        attention_mask : torch.Tensor
            List of indices specifying which tokens should be attended to by the model,
            by default None
            [Attention?](https://huggingface.co/transformers/glossary#attention-mask)

        Returns
        -------
        torch.Tensor
            Predicted labels for the classification problem
        """
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled_output = hidden_states[:, 0]
        y_hat = self.classifier(pooled_output)

        return y_hat

    def tokenize(self, sentences: Sequence[Union[str, list[str]]]) -> TensorDataset:
        """Convert sequence of sentences or tokens into DistilBERT inputs.

        Given a sequence of sentences or tokens, computes the ``input_ids``,
        and ``attention_masks`` and loads them on their respective
        tensor device. Any changes made to the tokenizer should be reflected here
        and the `.forward()` method.

        Parameters
        ----------
        sentences : Sequence[str | list[str]]
            Sequence of sentences or tokens to be transformed into inputs for BERT.

        Returns
        -------
        TensorDataset
            2 tensors representing ``input_ids`` and ``attention_masks``.
            For more in-depth on what each these represent:

            - **input_ids** -- List of token ids to be fed to a model.
                [Input IDs?](https://huggingface.co/transformers/glossary#input-ids)

            - **attention_mask** -- List of indices specifying which tokens should
                be attended to by the model (when `return_attention_mask=True` or if
                *"attention_mask"* is in `self.model_input_names`).
                [Mask?](https://huggingface.co/transformers/glossary#attention-mask)

            If using a non-DistilBert tokenizer, see the below. The token type ids
            aren't needed for DistilBert models.
            - **token_type_ids** -- List of token type ids to be fed to a model
                (when `return_token_type_ids=True` or if *"token_type_ids"* is in
                `self.model_input_names`).
                [Type IDs?](https://huggingface.co/transformers/glossary#token-type-ids)
        """
        sentences = [sent for sent in sentences]  # Input must be list
        batch_encoding = self.tokenizer.__call__(
            sentences,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return TensorDataset(batch_encoding.input_ids, batch_encoding.attention_mask)

    def fit(
        self,
        x_train: Dataset[Union[str, list[str]]],
        y_train: torch.Tensor,
        sample_weight: torch.Tensor = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.001,
    ):
        """Fit the model on the training data.

        Fine tunes a pre-trained BERT model on an input Sequence[str] by tokenizing the
        inputs and then fine tuning the last few layers of BERT and the classifier.

        Parameters
        ----------
        x_train : Dataset[str]
            Training data set of sentences or list[str] to be classified
        y_train : torch.Tensor
            Data Labels
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        batch_size : int, optional
            Training batch size, by default 2
        epochs : int, optional
            Number of training epochs, by default 1
        lr : float, optional
            Learning rate for the Model, by default 0.01

        Returns
        -------
        self : object
            Trained BERT classifier
        """
        if len(x_train) == 0:
            return self

        bert_inputs = self.tokenize(x_train)
        dataset = CatDataset(bert_inputs, y_train, sample_weight)

        # Optimizer and scheduler specified for BERT per Huggingface
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.1, total_iters=epochs * len(dataset)
        )
        criterion = F.cross_entropy

        self.train()
        for _ in range(int(epochs)):
            for input_batch, y_batch, *weights in DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=True
            ):
                input_batch = [t.to(self.bert.device) for t in input_batch]
                y_batch = y_batch.to(self.bert.device)

                optimizer.zero_grad()
                y_hat = self.__call__(*input_batch)

                if sample_weight is not None:
                    # F.cross_entropy doesn't support sample_weight
                    loss = criterion(y_hat, y_batch, reduction="none")
                    loss = (loss * weights[0].to(self.bert.device)).mean()
                else:
                    loss = criterion(y_hat, y_batch, reduction="mean")

                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

        return self

    def predict(self, x: Dataset[Union[str, list[str]]]):
        """Predict output from input sentences/tokens.

        Parameters
        ----------
        x : Dataset[str  |  list[str]]
            Input data set of sentences or list[str]

        Returns
        -------
        torch.Tensor
            Predicted labels as a tensor
        """
        if len(x) == 0:
            return torch.zeros(0, self.num_classes, device=self.bert.device)

        self.eval()
        # Return type of tokenizer is a data set so we are cheating here.
        bert_inputs = self.tokenize(x)
        bert_batch = [t.to(device=self.bert.device) for t in bert_inputs.tensors]

        y_hat = self.__call__(*bert_batch)
        return y_hat
