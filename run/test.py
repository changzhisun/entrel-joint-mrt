#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/02/01 17:32:24

@author: Changzhi Sun
"""
import os
import sys
sys.path.append('..')
import argparse
import json

import torch
import numpy as np

from lib import vocab, utils
from src import model
from entrel_eval import eval_file, eval_file_by_sample
from config import Configurable

torch.manual_seed(1) # CPU random seed
np.random.seed(1)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='../configs/default.cfg')
#  argparser.add_argument('--model', default='BaseParser')
args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)

use_cuda = config.use_cuda

# GPU and CPU using different random seed
if use_cuda:
    torch.cuda.manual_seed(1)

train_corpus = utils.load_entity_and_relation_sequences(config.train_file, sep="\t", schema=config.schema)
dev_corpus = utils.load_entity_and_relation_sequences(config.dev_file, sep="\t", schema=config.schema)
test_corpus = utils.load_entity_and_relation_sequences(config.test_file, sep="\t", schema=config.schema)

def load_json_file(filename):
    with open(filename, "r") as f:
        return json.load(f)

if config.parse_lstm_size != 0:
    train_lstm_h = load_json_file(config.parse_train_file)
    dev_lstm_h = load_json_file(config.parse_dev_file)
    test_lstm_h = load_json_file(config.parse_test_file)
else:
    train_lstm_h = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(len(train_corpus))]
    dev_lstm_h = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(len(dev_corpus))]
    test_lstm_h = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(len(test_corpus))]

def add_parse_to_copurs(corpus, lstm_h):
    for i in range(len(corpus)):
        corpus[i].append(lstm_h[i][:-1])
        if config.parse_lstm_size != 0:
            assert len(corpus[i][0]) == len(lstm_h[i]) - 1

add_parse_to_copurs(train_corpus, train_lstm_h)
add_parse_to_copurs(dev_corpus, dev_lstm_h)
add_parse_to_copurs(test_corpus, test_lstm_h)

max_sent_len = max([len(e[0]) for e in train_corpus + dev_corpus + test_corpus])
max_sent_len = min(max_sent_len, config.max_sent_len)

train_corpus = [e for e in train_corpus if len(e[0]) <= max_sent_len]
dev_corpus = [e for e in dev_corpus if len(e[0]) <= max_sent_len]
test_corpus = [e for e in test_corpus if len(e[0]) <= max_sent_len]

print("Total items in train corpus: %s" % len(train_corpus))
print("Total items in dev corpus: %s" % len(dev_corpus))
print("Total items in test corpus: %s" % len(test_corpus))
print("Max sentence length: %s" % max_sent_len)

word_vocab = vocab.Vocab("words", PAD="<PAD>", lower=True)
char_vocab = vocab.Vocab("chars", PAD="<p>", lower=False)
chunk_vocab = vocab.Vocab("chunk_tags", lower=False)
rel_vocab = vocab.Vocab("rel_tags", PAD="None", lower=False)
utils.create_vocab(train_corpus+dev_corpus+test_corpus,
                   [word_vocab, chunk_vocab],
                   char_vocab,
                   rel_vocab)

train_tensors = utils.data2tensors(train_corpus, [word_vocab, chunk_vocab], rel_vocab, char_vocab)
dev_tensors = utils.data2tensors(dev_corpus, [word_vocab, chunk_vocab], rel_vocab, char_vocab)
test_tensors = utils.data2tensors(test_corpus, [word_vocab, chunk_vocab], rel_vocab, char_vocab)

pretrained_embeddings = utils.load_word_vectors(config.pretrained_embeddings_file,
                                                config.word_dims,
                                                word_vocab)
char_embed_kwargs = {
        "vocab_size" : char_vocab.size,
        "embedding_size": config.char_dims,
        "out_channels" : config.char_output_channels,
        "kernel_sizes" : config.char_kernel_sizes
    }

word_char_embedding = model.WordCharEmbedding(
        word_vocab.size, config.word_dims, char_embed_kwargs,
        dropout=config.dropout, concat=True
    )
model.assign_embeddings(word_char_embedding.word_embeddings, pretrained_embeddings)
word_char_emb_dim = config.word_dims + config.char_output_channels * len(config.char_kernel_sizes)

mymodel = model.JointEntRelModel(word_char_embedding,
                                 word_char_emb_dim,
                                 config.rel_output_channels,
                                 config.rel_kernel_sizes,
                                 config.lstm_hiddens,
                                 config.parse_lstm_size,
                                 50,
                                 50,
                                 chunk_vocab.size,
                                 rel_vocab.size,
                                 max_sent_len,
                                 chunk_vocab,
                                 rel_vocab.PAD_ID,
                                 num_layers=config.lstm_layers,
                                 bidirectional=True,
                                 use_cuda=use_cuda,
                                 win=None,
                                 sch_k=config.schedule_k,
                                 dropout=config.dropout)
if use_cuda:
    mymodel.cuda()

if os.path.exists(config.load_model_path):
    state_dict = torch.load(
            open(config.load_model_path, 'rb'),
            map_location=lambda storage, loc: storage)
    mymodel.load_state_dict(state_dict)
    print("loading pre-trained model successful [%s]" % config.load_model_path)

def predict_all(tensors, batch_size):
    mymodel.eval()
    predictions = []
    new_tensors = []
    for i in range(0, len(tensors), batch_size):
        print("[ %d / %d ]" % (len(tensors), min(len(tensors), i + batch_size)))
        batch = tensors[i: i + batch_size]
        X, X_char, X_lstm_h, Y, Y_rel, X_len, X_mask, batch = utils.get_minibatch(batch, word_vocab, char_vocab)
        X = model.convert_long_variable(X, use_cuda)
        X_lstm_h = model.convert_float_variable(X_lstm_h, use_cuda)
        X_char = model.convert_long_variable(X_char, use_cuda)
        Y = model.convert_long_tensor(Y, use_cuda)
        X_mask = model.convert_float_variable(X_mask, use_cuda)
        new_tensors.extend(batch)
        _, _, pred_entity_tags, pred_rel_tags, _ = mymodel(X, X_char, X_lstm_h, X_len, X_mask, Y, Y_rel)
        predictions.extend(list(zip(pred_entity_tags, pred_rel_tags)))
    return predictions, new_tensors


batch_size = config.batch_size
for title, tensors in zip( ["train", "dev", "test"], [train_tensors, dev_tensors, test_tensors]):
    if title != "test" : continue
    print("\nEvaluating %s" % title)
    predictions, new_tensors = predict_all(tensors, config.batch_size)
    eval_path = os.path.join(config.save_dir, "final.%s.output" % title)
    utils.print_predictions(new_tensors,
                            predictions,
                            eval_path,
                            word_vocab,
                            chunk_vocab,
                            rel_vocab)
    eval_file(eval_path)
