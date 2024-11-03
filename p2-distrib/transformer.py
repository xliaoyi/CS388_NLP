# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, n_head, mask):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, d_model)
        self.w1 = nn.Linear(d_model, num_classes)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal, n_head, mask) for _ in range(num_layers)])

        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        embedded = self.emb(indices)
        x = self.pos_enc(embedded)

        attention_maps = []
        for layer in self.layers:
            x, attention_map = layer(x)
            attention_maps.append(attention_map)

        # output layer
        output = self.w1(x)
        logprob_matrix = torch.nn.functional.log_softmax(output, dim=-1)

        return logprob_matrix, attention_maps

        # raise Exception("Implement me")


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, n_head, mask):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.k = nn.Linear(self.head_dim, d_internal)
        self.q = nn.Linear(self.head_dim, d_internal)
        self.v = nn.Linear(self.head_dim, d_internal)

        self.w0 = nn.Linear(self.n_head * self.d_internal, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )
        self.mask = mask

        # raise Exception("Implement me")

    def forward(self, input_vecs):  # dim: (seq_len, d_model)
        # split into several heads
        split_input_vecs = input_vecs.reshape(input_vecs.shape[0], self.n_head,
                                              self.head_dim)  # dim: (seq_len, n_head, head_dim)

        K = self.k(split_input_vecs)  # dim: (seq_len, n_head, d_internal)
        K = K.permute(1, 0, 2)  # dim: (n_head, seq_len, d_internal)
        Q = self.q(split_input_vecs)
        Q = Q.permute(1, 0, 2)
        V = self.v(split_input_vecs)
        V = V.permute(1, 0, 2)

        # self-attention
        QK = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)  # dim: (n_head, seq_len, seq_len)
        softmax_QK = torch.nn.functional.softmax(QK, dim=-1)

        # mask
        if self.mask:
            tensor = torch.zeros(softmax_QK.shape)
            mask = torch.triu(torch.ones_like(tensor), diagonal=1)
            tensor.masked_fill_(mask.bool(), float('-inf'))

        QKV = torch.matmul(softmax_QK, V)  # dim: (n_head, seq_len, d_internal)
        QKV = QKV.permute(1, 0, 2)  # dim: (seq_len, n_head, d_internal)
        concatenated_heads = QKV.reshape(QKV.shape[0], self.n_head * self.d_internal)  # dim: (seq_len, d_model)
        ffn_input = self.w0(concatenated_heads)  # dim: (seq_len, d_model)

        # residual connection
        ffn_input += input_vecs

        # feedforward
        ffn_out = self.ffn(ffn_input)  # dim: (seq_len, d_model)

        # residual connection
        ffn_out += ffn_input

        return ffn_out, softmax_QK[0]


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # raise Exception("Not fully implemented yet")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(vocab_size=27, num_positions=20, d_model=256,
                        d_internal=64, num_classes=3, num_layers=1, n_head=8, mask=False)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            (log_probs, _) = model.forward(ex.input_tensor)
            loss = loss_fcn(log_probs, ex.output_tensor)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()

        print(f"Epoch {t}: loss = {loss_this_epoch}")
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False,
           do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
