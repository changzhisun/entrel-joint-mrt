#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/05 20:57:52

@author: Changzhi Sun
"""
import os
import json
import numpy as np

import torch
import torch.optim as optim


def load_sequences(filenames, sep=" ", col_ids=None, schema=None):
    sequences = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        with open(filename, encoding='utf-8') as fp:
            seq = []
            for line in fp:
                line = line.rstrip()
                if line:
                    line = line.split(sep)
                    if col_ids is not None:
                        line = [line[idx] for idx in col_ids]
                    seq.append(tuple(line))
                else:
                    if seq:
                        sequences.append(seq)
                    seq = []
            if seq:
                sequences.append(seq)
    return sequences


def load_entity_sequences(filenames, sep=" ", col_ids=None, schema="BIO"):
    def convert_sequence(source_path, target_path):
        fsource = open(source_path, "r", encoding="utf8")
        ftarget = open(target_path, "w", encoding="utf8")
        for line in fsource:
            sent = json.loads(line)
            tokens = sent['sentText'].split(' ')
            tags = ['O'] * len(tokens)
            for men in sent['entityMentions']:
                s, e = men['offset']
                if schema == "BIO":
                    tags[s] = 'B-' + men['label']
                    for j in range(s+1, e):
                        tags[j] = 'I-' + men['label']
                else:
                    if e - s == 1:
                        tags[s] = "U-" + men['label']
                    elif e - s == 2:
                        tags[s] = 'B-' + men['label']
                        tags[s+1] = 'E-' + men['label']
                    else:
                        tags[s] = 'B-' + men['label']
                        tags[e - 1] = 'E-' + men['label']
                        for j in range(s+1, e - 1):
                            tags[j] = 'I-' + men['label']
            for w, t in zip(tokens, tags):
                print("{0} {1}".format(w, t), file=ftarget)
            print(file=ftarget)
        fsource.close()
        ftarget.close()

    sequences = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        ent_filename = "entity.txt"
        target_filename = os.path.join(os.path.dirname(filename), ent_filename)
        convert_sequence(filename, target_filename)
        with open(target_filename, "r", encoding='utf-8') as fp:
            seq = []
            for line in fp:
                line = line.rstrip()
                if line:
                    line = line.split(sep)
                    if col_ids is not None:
                        line = [line[idx] for idx in col_ids]
                    seq.append(tuple(line))
                else:
                    if seq:
                        sequences.append(seq)
                    seq = []
            if seq:
                sequences.append(seq)
    return sequences


def load_entity_and_relation_sequences(filenames, sep="\t", schema="BIO"):
    def convert_sequence(source_path, target_path):
        fsource = open(source_path, "r", encoding="utf8")
        ftarget = open(target_path, "w", encoding="utf8")
        for line in fsource:
            sent = json.loads(line)
            tokens = sent['sentText'].split(' ')
            tags = ['O'] * len(tokens)
            id2ent = {}
            for men in sent['entityMentions']:
                id2ent[men['emId']] = men['offset']
                s, e = men['offset']
                if schema == "BIO":
                    tags[s] = 'B-' + men['label']
                    for j in range(s+1, e):
                        tags[j] = 'I-' + men['label']
                else:
                    if e - s == 1:
                        tags[s] = "U-" + men['label']
                    elif e - s == 2:
                        tags[s] = 'B-' + men['label']
                        tags[s+1] = 'E-' + men['label']
                    else:
                        tags[s] = 'B-' + men['label']
                        tags[e - 1] = 'E-' + men['label']
                        for j in range(s+1, e - 1):
                            tags[j] = 'I-' + men['label']
            for w, t in zip(tokens, tags):
                print("{0}\t{1}".format(w, t), file=ftarget)
            for men in sent['relationMentions']:
                em1_idx = id2ent[men['em1Id']]
                em2_idx = id2ent[men['em2Id']]
                em1_text = men['em1Text']
                em2_text = men['em2Text']
                direction = "-->"
                if em1_idx[0] > em2_idx[0]:
                    direction = "<--"
                    em1_idx, em2_idx = em2_idx, em1_idx
                    em1_text, em2_text = em2_text, em1_text
                label = men['label'] + direction
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(
                    em1_idx, em2_idx,
                    em1_text, em2_text,
                    label), file=ftarget)
            print(file=ftarget)
        fsource.close()
        ftarget.close()

    sequences = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        ent_rel_filename = "entity_relation.txt"
        target_filename = os.path.join(os.path.dirname(filename), ent_rel_filename)
        convert_sequence(filename, target_filename)
        with open(target_filename, "r", encoding='utf-8') as fp:
            seq = [[], []]
            for line in fp:
                line = line.rstrip()
                if line:
                    line = line.split(sep)
                    line = [line[idx] for idx in range(len(line))]
                    if len(line) == 2:
                        seq[0].append(tuple(line))
                    elif len(line) == 5:
                        seq[1].append((eval(line[0]), eval(line[1]), line[-1]))
                else:
                    if seq[0]:
                        sequences.append(seq)
                    seq = [[], []]
            if seq[0]:
                sequences.append(seq)
    return sequences


def load_word_vectors(vector_file, ndims, vocab):
    #  W = np.zeros((vocab.size, ndims), dtype="float32")
    W = np.random.uniform(-0.25, 0.25, (vocab.size, ndims))
    total, found = 0, 0
    with open(vector_file) as fp:
        for i, line in enumerate(fp):
            line = line.rstrip().split()
            if line:
                total += 1
                try:
                    assert len(line) == ndims+1,(
                        "Line[{}] {} vector dims {} doesn't match ndims={}".format(i, line[0], len(line)-1, ndims)
                    )
                except AssertionError as e:
                    print(e)
                    continue
                word = line[0]
                idx = vocab.getidx(word)
                if idx >= vocab.offset:
                    found += 1
                    vecs = np.array(list(map(float, line[1:])))
                    W[idx, :] = vecs
    # Write to cache file
    print("Found {} [{:.2f}%] vectors from {} vectors in {} with ndims={}".format(
        found, found * 100/vocab.size, total, vector_file, ndims))
    #  norm_W = np.sqrt((W*W).sum(axis=1, keepdims=True))
    #  valid_idx = norm_W.squeeze() != 0
    #  W[valid_idx, :] /= norm_W[valid_idx]
    return W

def load_bin_vec(vector_file, ndims, vocab, cache_file, override_cache=False):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    if os.path.exists(cache_file) and not override_cache:
        W = np.load(cache_file)
        return W
    word_vecs = {}
    with open(vector_file, "rb") as f:
        header = f.readline()
        length = vocab.size
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            i = 0
            if len(word_vecs) == length:
                return word_vecs
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            word = word.decode()
            if word in vocab.item2idx:
                i += 1
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    found = len(word_vecs)
    total = vocab_size

    W = np.random.uniform(-0.25, 0.25, (vocab.size, ndims))
    for word in vocab.item2idx:
        index = vocab.item2idx[word]
        if index == 0:
            continue
        if word in word_vecs:
            W[index] = word_vecs[word]
    print("Found {} [{:.2f}%] vectors from {} vectors in {} with ndims={}".format(
        found, found * 100/vocab.size, total, vector_file, ndims))
    print("Caching embedding with shape {} to {}".format(W.shape, cache_file))
    np.save(cache_file, W)
    return W


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = [data[e] for e in shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def hyperparam_string(config):
    """Hyerparam string."""
    exp_name = ''
    exp_name += 'model_%s__' % (config['data']['corpus'])
    exp_name += 'hidden_dim_%s__' % (config['model']['hidden_dim'])
    exp_name += 'word_emb_dim_%s__' % (config['model']['word_emb_dim'])
    exp_name += 'optimizer_%s__' % (config['training']['optimizer'])
    exp_name += 'num_layers_%d__' % (config['model']['num_layers'])
    exp_name += 'bidir_%s' % (config['model']['bidirectional'])

    return exp_name


def get_minibatch(batch, word_vocab, char_vocab):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    X_len = [len(s[0]) for s in batch]
    max_batch_sent_len = max(X_len)
    max_batch_char_len = max([len(c) for s in batch for c in s[3]])

    X = []
    X_char = []
    X_lstm_h = []
    Y = []
    Y_rel = []
    for s in batch:
        X.append(s[0] + [word_vocab.PAD_ID] * (max_batch_sent_len - len(s[0])))
        lstm_h_zero = [0.0 for _ in range(len(s[2][0]))]
        X_lstm_h.append(s[2] + [lstm_h_zero] * (max_batch_sent_len - len(s[2])))
        char_pad = []
        for c in s[3]:
            char_pad.append(c + [char_vocab.PAD_ID] * (max_batch_char_len - len(c)))
        X_char.append(char_pad + [[char_vocab.PAD_ID] * max_batch_char_len] * (max_batch_sent_len - len(s[0])))
        Y.append(s[1] + [-1] * (max_batch_sent_len - len(s[1])))
        Y_rel.append(s[4])
        assert len(X[-1]) == len(Y[-1])
        assert len(X[-1]) == len(X_char[-1])
    for i in range(len(batch)):
        assert batch[i][0] == X[i][:X_len[i]]
        assert batch[i][1] == Y[i][:X_len[i]]
        assert batch[i][4] == Y_rel[i]
        for j in range(X_len[i]):
            k = 0
            while k < len(X_char[i][j]):
                if X_char[i][j][k] == char_vocab.PAD_ID:
                    break
                k += 1
            assert batch[i][3][j] == X_char[i][j][:k]
        for e in X_char[i]:
            assert len(e) == max_batch_char_len
        assert len(X[i]) == max_batch_sent_len
        assert len(Y[i]) == max_batch_sent_len
        assert len(X_char[i]) == max_batch_sent_len
    X = np.array(X)
    X_char = np.array(X_char)
    X_lstm_h = np.array(X_lstm_h)
    Y = np.array(Y)
    mask = [[1.0] * l  + [0.0] * (max_batch_sent_len - l) for l in X_len]
    return X, X_char, X_lstm_h, Y, Y_rel, X_len, mask, batch

def create_vocab(data, vocabs, char_vocab, rel_vocab, word_idx=0):
    n_vocabs = len(vocabs)
    for sent in data:
        for token_tags in sent[0]:
            for vocab_id in range(n_vocabs):
                vocabs[vocab_id].add(token_tags[vocab_id])
            char_vocab.batch_add(token_tags[word_idx])
        for rel_tags in sent[1]:
            rel_vocab.add(rel_tags[-1])
    print("Created vocabs: %s, relation[%s], chars[%s]" % (", ".join(
        "{}[{}]".format(vocab.name, vocab.size)
        for vocab in vocabs
    ), rel_vocab.size, char_vocab.size))

def data2tensors(data, vocabs, rel_vocab, char_vocab, word_idx=0, column_ids=(0, -1), win=15):
    vocabs = [vocabs[idx] for idx in column_ids]
    n_vocabs = len(vocabs)
    tensors = []
    for sent in data:
        sent_vecs = [[] for i in range(n_vocabs+3)] # Last two are for char and relation vecs
        char_vecs = []
        for token_tags in sent[0]:
            vocab_id = 0 # First column is the word
            # lowercase the word
            sent_vecs[vocab_id].append(
                    vocabs[vocab_id].getidx(token_tags[vocab_id].lower())
                )
            for vocab_id in range(1, n_vocabs):
                sent_vecs[vocab_id].append(
                    vocabs[vocab_id].getidx(token_tags[vocab_id])
                )
            sent_vecs[-2].append(
                [char_vocab.getidx(c) for c in token_tags[word_idx]]
            )
        for b, e, t in sent[1]:
            sent_vecs[-1].append([b, e, rel_vocab.getidx(t)])
        for token_lstm_h in sent[2]:
            sent_vecs[-3].append(token_lstm_h)
        tensors.append(sent_vecs)
    return tensors

def print_predictions(corpus,
                      predictions,
                      filename,
                      word_vocab,
                      chunk_vocab,
                      rel_vocab):
    with open(filename, "w+") as fp:
        i = 0
        for seq, pred in zip(corpus, predictions):
            i += 1
            seq_len = len(seq[0])
            assert len(seq[0]) == len(pred[0])
            assert len(seq[1]) == len(pred[0])
            for (idx, true_label), pred_label in zip(zip(seq[0], seq[1]), pred[0]):
                pred_label = chunk_vocab.idx2item[pred_label]
                token = word_vocab.idx2item[idx]
                true_label = chunk_vocab.idx2item[true_label]
                print("{}\t{}\t{}".format(token, true_label, pred_label), file=fp)
            for s, e, r in seq[4]:
                r = rel_vocab.idx2item[r]
                if r[-3:] == "<--":
                    s, e = e, s
                r = r[:-3]
                print("Rel-True\t{}\t{}\t{}".format(s, e, r), file=fp)
            for s, e, r in pred[1]:
                r = rel_vocab.idx2item[r]
                assert int(s[-1]) < len(pred[0])
                assert int(e[-1]) < len(pred[0])
                s = [s[0], s[-1] + 1]
                e = [e[0], e[-1] + 1]
                if r[-3:] == "<--":
                    s, e = e, s
                r = r[:-3]
                print("Rel-Pred\t{}\t{}\t{}".format(s, e, r), file=fp)
            print(file=fp) # Add new line after each sequence
