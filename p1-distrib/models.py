# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import pandas as pd


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):

        self.indexer = indexer

        train_exs = read_sentiment_examples("data/train.txt")
        # dev_exs = read_sentiment_examples("data/dev.txt")

        for ex in train_exs:
            for word in ex.words:
                self.indexer.add_and_get_index(word)

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> List[int]:

        # counter = Counter()
        vet = [0] * self.indexer.objs_to_ints.__len__()

        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
                vet[idx] += 1

            if self.indexer.contains(word):
                vet[self.indexer.index_of(word)] += 1

        return vet

        # raise Exception("Must be implemented")


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        train_exs = read_sentiment_examples("data/train.txt")

        for ex in train_exs:
            for i in range(len(ex.words) - 1):
                self.indexer.add_and_get_index(ex.words[i] + ex.words[i + 1])

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> List[int]:
        vet = [0] * self.indexer.objs_to_ints.__len__()
        for i in range(len(sentence) - 1):
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(sentence[i] + sentence[i + 1])
                vet[idx] += 1

            if self.indexer.contains(sentence[i] + sentence[i + 1]):
                vet[self.indexer.index_of(sentence[i] + sentence[i + 1])] += 1

        return vet


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.freq = dict()
        train_exs = read_sentiment_examples("data/train.txt")

        for ex in train_exs:
            for i in range(len(ex.words)):
                idx = self.indexer.add_and_get_index(ex.words[i])
                if idx not in self.freq:
                    self.freq[idx] = 1
                else:
                    self.freq[idx] += 1

    # def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> List[int]:
    #     vet = [0] * self.indexer.objs_to_ints.__len__()
    #     for i in range(len(sentence) - 1):
    #         if add_to_indexer:
    #             idx = self.indexer.add_and_get_index(sentence[i] + sentence[i + 1])
    #             vet[idx] += 1
    #
    #         if self.indexer.contains(sentence[i] + sentence[i + 1]) and self.freq[self.indexer.index_of(sentence[i] + sentence[i + 1])] > 2:
    #             vet[self.indexer.index_of(sentence[i] + sentence[i + 1])] += 1
    #
    #     return vet
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> List[int]:
        vet = [0] * self.indexer.objs_to_ints.__len__()
        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
                vet[idx] += 1

            if self.indexer.contains(word) and self.freq[self.indexer.index_of(word)] > 1:
                vet[self.indexer.index_of(word)] += 1

        return vet


class LogisticRegressionClassifier(SentimentClassifier, nn.Module):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    # def __init__(self):
    #     raise Exception("Must be implemented")
    def __init__(self, input_dim):
        super(LogisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.feature_extractor = UnigramFeatureExtractor(Indexer())
        # self.feature_extractor = BigramFeatureExtractor(Indexer())
        # self.feature_extractor = BetterFeatureExtractor(Indexer())

    def forward(self, x):
        # Apply the linear layer and then a sigmoid to get the probability
        return torch.sigmoid(self.linear(x))

    def predict(self, ex_words: List[str]) -> int:
        features = self.feature_extractor.extract_features(ex_words, add_to_indexer = False)
        features_tensor = torch.tensor(features, dtype = torch.float32).unsqueeze(0)
        output = self.forward(features_tensor)
        label = 1 if output.item() >= 0.5 else 0
        return label


def train_logistic_regression(train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # shuffle the list
    random.seed(0)
    random.shuffle(train_exs)

    feat_matrix = []
    labels = []
    # extract features
    for sentence in train_exs:
        feat = feat_extractor.extract_features(sentence = sentence.words, add_to_indexer = False)
        lbl = sentence.label
        feat_matrix.append(feat)
        labels.append(lbl)

    # convert to numpy array
    feat_matrix = np.array(feat_matrix)
    labels = np.array(labels)

    feat_matrix_tensor = torch.tensor(feat_matrix, dtype = torch.float32)
    labels_tensor = torch.tensor(labels, dtype = torch.float32)

    # train the model
    input_dim = feat_matrix_tensor.shape[1]
    print(input_dim)
    model = LogisticRegressionClassifier(input_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr = 0.03)
    learning_rate = 0.03

    # # record
    # epoch_list = []
    # train_acc = []
    # dev_acc = []

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(feat_matrix_tensor)
        loss = criterion(outputs.squeeze(), labels_tensor)
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param -= learning_rate * param.grad
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        # predict

        # # train accuracy
        # pred_results = []
        # gold_results = [i.label for i in train_exs]
        # for sentence in train_exs:
        #     pred = model.predict(sentence.words)
        #     pred_results.append(pred)
        # correct = 0
        # for i in range(len(pred_results)):
        #     if pred_results[i] == gold_results[i]:
        #         correct += 1
        # print(f'Train Accuracy: {correct / len(pred_results)}')
        # train_acc.append(correct / len(pred_results))
        #
        # # dev accuracy
        # pred_results = []
        # gold_results = [i.label for i in dev_exs]
        # for sentence in dev_exs:
        #     pred = model.predict(sentence.words)
        #     pred_results.append(pred)
        # correct = 0
        # for i in range(len(pred_results)):
        #     if pred_results[i] == gold_results[i]:
        #         correct += 1
        # print(f'Dev Accuracy: {correct / len(pred_results)}')
        # dev_acc.append(correct / len(pred_results))
        #
        # epoch_list.append(epoch + 1)

    # save
    # pd.DataFrame({'epoch': epoch_list, 'train_acc': train_acc, 'dev_acc': dev_acc}).to_csv('Adagrad_log.csv',
    #                                                                                        index = False)

    return model


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, dev_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, input_dim, hidden_dim, n_classes, word_embeddings):
        super(NeuralSentimentClassifier, self).__init__()
        self.word_embeddings = word_embeddings
        self.V = nn.Linear(input_dim, hidden_dim)
        self.g = nn.ReLU()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.g = nn.ReLU()
        self.WW = nn.Linear(hidden_dim, n_classes)
        self.log_softmax = nn.LogSoftmax(dim = 0)

        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

        # raise NotImplementedError

    def forward(self, x):
        return torch.sigmoid(self.WW(self.g(self.W(self.g(self.V(x))))))

    def predict(self, ex_words: List[str]) -> int:
        word_matrix = []
        for word in ex_words:
            word_matrix.append(self.word_embeddings.get_embedding(word))
        word_matrix = np.array(word_matrix)
        feat = np.mean(word_matrix, axis = 0)
        features_tensor = torch.tensor(feat, dtype = torch.float32).unsqueeze(0)
        output = self.forward(features_tensor)

        label = 1 if output[0][0] < output[0][1] else 0
        return label


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    word_embeddings = read_word_embeddings(args.word_vecs_path)

    random.seed(0)
    random.shuffle(train_exs)

    feat_matrix = []
    labels = []

    # extract features
    for sentence in train_exs:
        word_matrix = []
        for word in sentence.words:
            word_matrix.append(word_embeddings.get_embedding(word))
        word_matrix = np.array(word_matrix)
        feat = np.mean(word_matrix, axis = 0)
        lbl = sentence.label
        feat_matrix.append(feat)
        labels.append(lbl)

    # convert to numpy array
    feat_matrix = np.array(feat_matrix)
    labels = np.array(labels)
    num_categories = np.max(labels) + 1
    one_hot_encoded = np.eye(num_categories)[labels]

    feat_matrix_tensor = torch.tensor(feat_matrix, dtype = torch.float32)
    labels_tensor = torch.tensor(one_hot_encoded, dtype = torch.float32)

    # train the model
    input_dim = feat_matrix_tensor.shape[1]
    hidden_dim = 256
    n_classes = 2

    model = NeuralSentimentClassifier(input_dim, hidden_dim, n_classes, word_embeddings)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0003)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(feat_matrix_tensor)
        # print(f"outputs: {outputs}")
        # print(f"labels: {labels_tensor}")
        loss = criterion(outputs.squeeze(), labels_tensor)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model
    # raise NotImplementedError
