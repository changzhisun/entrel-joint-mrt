#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/05 21:15:15

@author: Changzhi Sun
"""
import os
import sys
sys.path.append('..')
import argparse
import json

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from lib import vocab, utils
from src import model
from entrel_eval import eval_file, eval_file_by_sample
from config import Configurable

torch.manual_seed(1) # CPU random seed
np.random.seed(1)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='../configs/default.cfg')
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
    print("loading previous model successful [%s]" % config.load_model_path)

parameters = [p for p in mymodel.parameters() if p.requires_grad]
optimizer = optim.Adadelta(parameters)

def step(batch):
    X, X_char, X_lstm_h, Y, Y_rel, X_len, X_mask, batch = utils.get_minibatch(batch, word_vocab, char_vocab)
    X = model.convert_long_variable(X, use_cuda)
    X_char = model.convert_long_variable(X_char, use_cuda)
    X_lstm_h = model.convert_float_variable(X_lstm_h, use_cuda)
    Y = model.convert_long_tensor(Y, use_cuda)
    X_mask = model.convert_float_variable(X_mask, use_cuda)
    entity_loss, relation_loss, pred_entity_tags, pred_rel_tags, candi_rel_num = mymodel(X, X_char, X_lstm_h, X_len, X_mask, Y, Y_rel, i)
    return entity_loss, relation_loss, pred_entity_tags, pred_rel_tags, X_len, candi_rel_num, batch


def train_step(batch, optimizer):
    optimizer.zero_grad()
    mymodel.train()
    mymodel.rel_model.sampling = False
    entity_loss, relation_loss, _, _, X_len, candi_rel_num, _ = step(batch)
    if candi_rel_num == 0:
        relation_loss = model.convert_float_variable([0], use_cuda)
        relation_loss.requires_grad = True
    else:
        relation_loss = relation_loss / candi_rel_num
    entity_loss = entity_loss / sum(X_len)
    loss = entity_loss + relation_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, config.clip_c)
    optimizer.step()
    print('Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f)' % (
        i, j, loss.item(), entity_loss.item(), relation_loss.item()))


def dev_step(dev_tensors, batch_size):
    optimizer.zero_grad()
    mymodel.eval()
    predictions = []
    entity_losses = []
    relation_losses = []
    new_tensors = []
    all_ent_num = 0
    all_rel_num = 0
    for k in range(0, len(dev_tensors), batch_size):
        batch = dev_tensors[k: k + batch_size]
        entity_loss, relation_loss, pred_entity_tags, pred_rel_tags, X_len, candi_rel_num, batch = step(batch)
        all_rel_num += candi_rel_num
        all_ent_num += sum(X_len)
        predictions.extend(list(zip(pred_entity_tags, pred_rel_tags)))
        entity_losses.append(entity_loss.item())
        relation_losses.append(relation_loss.item())
        new_tensors.extend(batch)
    entity_loss = sum(entity_losses) / all_ent_num
    if all_rel_num == 0:
        relation_loss = 0
    else:
        relation_loss = sum(relation_losses) / all_rel_num
    loss = entity_loss + relation_loss

    print('Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f)' % (
        i, j, loss, entity_loss, relation_loss))

    eval_path = os.path.join(config.save_dir, "validate.dev.output")
    utils.print_predictions(new_tensors,
                            predictions,
                            eval_path,
                            word_vocab,
                            chunk_vocab,
                            rel_vocab)
    entity_score, relation_score = eval_file(eval_path)
    return relation_score


batch_size = config.batch_size
best_f1 = 0
for i in range(config.train_iters):
    np.random.shuffle(train_tensors)

    for j in range(0, len(train_tensors), batch_size):
        batch = train_tensors[j: j + batch_size]

        train_step(batch, optimizer)

        if j > 0 and j % config.validate_every == 0:

            print('Evaluating model in dev set...')

            dev_f1 = dev_step(dev_tensors, batch_size)

            if dev_f1 > best_f1:

                best_f1 = dev_f1
                print('Saving model ...')
                torch.save(mymodel.state_dict(),
                    open(os.path.join(config.save_dir, "minibatch", 'epoch__%d__minibatch_%d' % (i, j)), "wb"))
                torch.save(mymodel.state_dict(), open(config.save_model_path, "wb"))
