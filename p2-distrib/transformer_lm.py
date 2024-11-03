# models.py
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import PositionalEncoding, TransformerLayer, Transformer


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, voc_size, d_model, nhead, dim_feedforward, num_layers, vocab_index):
        super().__init__()
        self.voc_size = voc_size
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
                d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward, batch_first = True
                )
        self.transformer = torch.nn.TransformerEncoder(self.transformer_layer, num_layers = num_layers)

        self.emb = nn.Embedding(voc_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, d_model)
        self.linear = nn.Linear(d_model, voc_size)
        self.vocab_index = vocab_index

    def forward(self, context_list):
        indices = [[self.vocab_index.index_of(c) for c in context] for context in context_list]
        # indices = [self.vocab_index.index_of(c) for c in context]
        indices = torch.LongTensor(indices) # (seq_len)
        # embed the context
        embedded = self.emb(indices) # (seq_len, d_model)
        x = self.pos_enc(embedded) # (seq_len, d_model)

        # build the mask
        tensor = torch.zeros(indices.shape[-1], indices.shape[-1])
        mask = torch.triu(torch.ones_like(tensor), diagonal = 1)
        tensor.masked_fill_(mask.bool(), float('-inf'))

        # pass through the transformer
        x = self.transformer(x, mask = mask)  # (batch_size, seq_len, d_model)

        # get the last character
        x = x[:, -1, :] # (batch_size, d_model)
        x = self.linear(x)
        return F.log_softmax(x, dim = -1)

    def get_next_char_log_probs(self, context):
        if len(context) == 0:
            return np.log(np.ones([self.voc_size]) / self.voc_size)
        else:
            log_probs = self.forward([context])[0]
            return log_probs.detach().numpy()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # raise Exception("Implement me")
    d_model = 128
    nhead = 8
    dim_feedforward = 128
    num_layers = 2
    lr = 1e-4
    epochs = 5
    batch_size = 8
    voc_size = 27

    # load and get indices for the train and dev text
    texts = []
    # for i in range(0, len(train_text)):
    #     text = train_text[max(0, i - 20):i]
    #     if len(text) < 2:
    #         continue
    #     else:
    #         texts.append(text)
    for i in range(0, len(train_text)):
        text = train_text[max(0, i - 20):i]
        if len(text) != 20:
            continue
        else:
            texts.append(text)

    # texts = texts[:4]
    random.seed(0)
    random.shuffle(texts)
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    for char in vocab:
        vocab_index.add_and_get_index(char)

    texts_x = [i[:-1] for i in texts]
    texts_y = [i[-1:] for i in texts]
    texts_y = [vocab_index.index_of(i) for i in texts_y]
    texts_y = torch.LongTensor(texts_y)

    # create the model
    model = NeuralLanguageModel(voc_size = voc_size, d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward,
                                num_layers = num_layers, vocab_index = vocab_index)
    model.zero_grad()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        loss_this_epoch = 0.0
        loss_fcn = nn.NLLLoss()
        for i in range(0, len(texts), batch_size):
            x = texts_x[i:i + batch_size]
            y = texts_y[i:i + batch_size]

            # if y == -1:
            #     continue

            log_probs = model.forward(x)

            loss = loss_fcn(log_probs, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()

        print(f"Epoch {epoch}: loss = {loss_this_epoch}")
    model.eval()
    return model
