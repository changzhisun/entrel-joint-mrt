#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/07/13 14:16:42

@author: Changzhi Sun
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import defaultdict
import re

def convert_long_tensor(var, use_cuda):
    var = torch.LongTensor(var)
    if use_cuda:
        var = var.cuda(async=True)
    return var

def convert_float_tensor(var, use_cuda):
    var = torch.FloatTensor(var)
    if use_cuda:
        var = var.cuda(async=True)
    return var

def convert_long_variable(var, use_cuda):
    return Variable(convert_long_tensor(var, use_cuda))

def convert_float_variable(var, use_cuda):
    return Variable(convert_float_tensor(var, use_cuda))

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp_torch(vecs, axis=None):
    ## Use help from: http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    if axis < 0:
        axis = vecs.ndimension()+axis
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.expand_as(vecs)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    #print(max_val, out_val)
    return max_val + out_val

def assign_embeddings(embedding_module, pretrained_embeddings, fix_embedding=False):
    embedding_module.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
    if fix_embedding:
        embedding_module.weight.requires_grad = False

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'U': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'U': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'U': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'U': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'U' and tag == 'E': chunk_start = True
    if prev_tag == 'U' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

class CharEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 out_channels,
                 kernel_sizes,
                 padding_idx=0,
                 dropout=0.5):
        super(CharEmbedding, self).__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        # Usage of nn.ModuleList is important
        ## See: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/6
        self.convs1 = nn.ModuleList([nn.Conv2d(1, out_channels, (K, embedding_size), padding=(K-1, 0))
                                     for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        x = self.char_embeddings(X)
        x = self.dropout(x)
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return self.dropout(x)


class WordCharEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 char_embed_kwargs,
                 dropout=0.5,
                 aux_embedding_size=None,
                 padding_idx=0,
                 concat=False):
        super(WordCharEmbedding, self).__init__()
        self.char_embeddings = CharEmbedding(**char_embed_kwargs)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        if concat and aux_embedding_size is not None:
            ## Only allow aux embedding in concat mode
            self.aux_word_embeddings = nn.Embedding(vocab_size, aux_embedding_size)
        self.concat = concat

    def forward(self, X, X_char=None):
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        word_vecs = self.word_embeddings(X)
        if X_char is not None:
            #  char_vecs = torch.cat([
                #  self.char_embeddings(x).unsqueeze(0)
                #  for x in X_char
            #  ], 1)
            batch_size, sent_size, char_size = X_char.size()
            X_char = X_char.view(-1, char_size)
            char_vecs = self.char_embeddings(X_char)
            char_vecs = char_vecs.view(batch_size, sent_size, -1)
            if self.concat:
                embedding_list = [char_vecs, word_vecs]
                if hasattr(self, "aux_word_embeddings"):
                    aux_vecs = self.aux_word_embeddings(X)
                    embedding_list.append(aux_vecs)
                word_vecs = torch.cat(embedding_list, 2)
            else:
                word_vecs = char_vecs + word_vecs
        return self.dropout(word_vecs)


class EntModel(nn.Module):

    def __init__(self,
                 word_char_embedding,
                 word_char_emb_size,
                 hidden_size,
                 parse_lstm_size,
                 tag_emb_size,
                 tag_size,
                 chunk_vocab,
                 num_layers=1,
                 use_cuda=False,
                 bidirectional=True,
                 dropout=0.5):
        super(EntModel, self).__init__()
        self.word_char_embedding = word_char_embedding
        self.word_char_emb_size = word_char_emb_size
        self.use_cuda = use_cuda
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.parse_lstm_size = parse_lstm_size
        self.tag_size = tag_size
        self.chunk_vocab = chunk_vocab
        self.dropout = nn.Dropout(dropout)
        self.tag_emb_size = tag_emb_size

        self.chunk_num, self.idx2chunk = self.parse_chunk_vocab(chunk_vocab)
        self.chunk2idx = {v : k for k, v in enumerate(self.idx2chunk)}

        self.tag_embeddings = nn.Embedding(tag_size, tag_emb_size)

        self.encoder = nn.LSTM(word_char_emb_size + self.parse_lstm_size,
                               self.hidden_size // 2,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True,
                               dropout=dropout)
        self.output_seq = nn.Sequential(nn.Tanh(),
                                        self.dropout,
                                        nn.Linear(self.hidden_size + tag_emb_size, tag_size))
        self.decoder_output = nn.Linear(hidden_size, tag_size)

        self.loss_function = nn.CrossEntropyLoss(size_average=False)
        self.Softmax= nn.Softmax(dim=1)

    def parse_chunk_vocab(self, chunk_vocab):
        entity_set = set()
        for tag in chunk_vocab.idx2item:
            if tag == 'O':
                continue
            _, tag_type = parse_tag(tag)
            entity_set.add(tag_type)
        entity_id2item = list(entity_set)
        return len(entity_id2item), entity_id2item

    def forward(self, X, X_char, X_lstm_h, X_len, X_mask, Y):
        batch_size = X.size(0)
        seq_size = X.size(1)
        batch_wc_embs, batch_rnn_outputs = self.run_rnn(X, X_char, X_lstm_h, X_len, X_mask)  # batch_size x seq_size x hidden_size
        assert batch_rnn_outputs.size() == (batch_size, seq_size, self.hidden_size)

        batch_emissions = self.get_emissions(batch_rnn_outputs)
        assert batch_emissions.size() == (batch_size, seq_size, self.tag_size)

        batch_ent_loss, batch_pred_ent_tags = self.get_loss_and_predict(batch_emissions, Y, X_len)
        return batch_ent_loss, batch_pred_ent_tags, batch_wc_embs, batch_rnn_outputs

    def run_rnn(self, X, X_char, X_lstm_h, X_len, X_mask):
        batch_size = X.size(0)
        seq_size = X.size(1)
        embed = self.word_char_embedding(X, X_char) # batch_size x seq_size x embed_size

        assert embed.size() == (batch_size, seq_size, self.word_char_emb_size)

        if self.parse_lstm_size != 0:
            embed = torch.cat([embed, X_lstm_h], 2)
            #  print(embed.size())

        embed_pack = nn.utils.rnn.pack_padded_sequence(embed, X_len, batch_first=True)
        encoder_outputs, _ = self.encoder(embed_pack) # batch_size x seq_size x hidden_size
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True) # batch_size x seq_size x hidden_size
        encoder_outputs = self.dropout(encoder_outputs)
        return embed, encoder_outputs

    def get_emissions(self, batch_rnn_outputs):
        batch_size = batch_rnn_outputs.size(0)
        batch_rnn_outputs = batch_rnn_outputs.contiguous()
        batch_emissions = self.decoder_output(batch_rnn_outputs.view(-1, self.hidden_size)) # (batch_size * seq_size) x tag_size
        batch_emissions = batch_emissions.view(batch_size, -1, self.tag_size) # batch_size x seq_size x tag_size
        return batch_emissions

    def get_loss_and_predict(self, batch_emissions, Y, X_len):
        losses = []
        pred_tags = []
        for i, emissions in enumerate(batch_emissions):
            emissions = emissions[:X_len[i]]
            y = Variable(Y[i][:X_len[i]])

            loss = self.loss_function(emissions, y)

            _, pred_tag = emissions.max(1)
            if self.use_cuda:
                pred_tag = pred_tag.cpu()

            pred_tag = [e.item() for e in pred_tag]
            pred_tags.append(pred_tag)
            losses.append(loss.unsqueeze(0))

            assert len(pred_tags[-1]) == X_len[i]
            assert len(pred_tags[-1]) ==len(y)
        batch_ent_loss = torch.sum(torch.cat(losses, 0))
        return batch_ent_loss, pred_tags

    def get_prob(self, X, X_char, X_lstm_h, X_len, X_mask, Y):
        batch_size = X.size(0)
        seq_size = X.size(1)

        batch_wc_embs, batch_rnn_outputs = self.run_rnn(X, X_char, X_lstm_h, X_len, X_mask)  # batch_size x seq_size x hidden_size
        assert batch_rnn_outputs.size() == (batch_size, seq_size, self.hidden_size)

        batch_emissions = self.get_emissions(batch_rnn_outputs)
        assert batch_emissions.size() == (batch_size, seq_size, self.tag_size)

        prob_list = []
        for i, emissions in enumerate(batch_emissions):
            emissions = emissions[:X_len[i]]
            probs = self.Softmax(emissions)
            prob_list.append(probs)
        return prob_list

    def sample_by_prob(self, probs, epsilon=0.1):
        assert self.tag_size == probs.size(1)
        sample_y = probs.multinomial(1).data.squeeze(1).tolist()
        for i in range(len(sample_y)):
            rd = np.random.random()
            if rd < epsilon:
              sample_y[i] = np.random.randint(0, probs.size(1))
        return sample_y

class RelModel(nn.Module):

    def __init__(self,
                 ent_model,
                 N_ID,
                 out_channels,
                 kernel_sizes,
                 relation_size,
                 position_emb_size,
                 max_sent_len,
                 use_cuda=False,
                 dropout=0.5,
                 win=15):
        print("win=", win)
        super(RelModel, self).__init__()
        self.ent_model = ent_model
        self.N_ID = N_ID
        self.max_sent_len = max_sent_len
        self.win = win
        self.relation_size = relation_size
        self.use_cuda = use_cuda
        self.sampling = False

        self.position_embeddings = nn.Embedding(2 * max_sent_len, position_emb_size)

        self.dropout = nn.Dropout(dropout)
        #  self.conv_input_size = self.ent_model.hidden_size + self.ent_model.tag_emb_size + self.ent_model.word_char_emb_size
        self.conv_input_size = self.ent_model.hidden_size + self.ent_model.tag_size

        self.b_convs = nn.ModuleList([nn.Conv2d(1,
                                                out_channels,
                                                (K, self.conv_input_size),
                                                padding=(K-1, 0))
                                      for K in kernel_sizes])
        self.mid_convs = nn.ModuleList([nn.Conv2d(1,
                                                  out_channels,
                                                  #  (K, self.conv_input_size - self.ent_model.tag_size),
                                                  (K, self.conv_input_size),
                                                  padding=(K-1, 0))
                                        for K in kernel_sizes])
        self.e_convs = nn.ModuleList([nn.Conv2d(1,
                                                out_channels,
                                                (K, self.conv_input_size),
                                                padding=(K-1, 0))
                                      for K in kernel_sizes])

        self.relation_input_size = len(kernel_sizes) * out_channels * 3 + max_sent_len + 2 * self.ent_model.hidden_size
        #  self.relation_input_size = len(kernel_sizes) * out_channels * 3 + max_sent_len
        self.Tanh = nn.Tanh()
        self.Softmax= nn.Softmax(dim=1)

        self.output_seq = nn.Sequential(nn.Linear(self.relation_input_size,
                                                  self.ent_model.hidden_size),
                                        nn.ReLU(),
                                        self.dropout,
                                        nn.Linear(self.ent_model.hidden_size,
                                                  relation_size))
        self.loss_function = nn.CrossEntropyLoss(size_average=False)
        #  self.loss_function = nn.MultiMarginLoss()
        self.ent_emb_oh = nn.Embedding(self.ent_model.tag_size, self.ent_model.tag_size)
        assign_embeddings(self.ent_emb_oh, np.eye(self.ent_model.tag_size), fix_embedding=True)

    def forward(self,
                batch_wc_embs,
                batch_rnn_outputs,
                batch_pred_ent_tags,
                Y_rel,
                X_len):
        batch_size = len(X_len)
        candi_ent_idxs, labels, idx2batch = self.generate_candidate_entity_pair(batch_pred_ent_tags, Y_rel, X_len)
        if len(candi_ent_idxs) == 0:
            loss = convert_float_variable([0], self.use_cuda)
            loss = loss.squeeze(0)
            loss.requires_grad = True
            return loss, [[] for _ in range(batch_size)], 0

        scores = self.get_score(candi_ent_idxs,
                                idx2batch,
                                batch_pred_ent_tags,
                                batch_wc_embs,
                                batch_rnn_outputs,
                                X_len)
        batch_rel_loss, batch_pred_rel_tags = self.get_loss_and_predict(scores,
                                                                        labels,
                                                                        batch_size,
                                                                        idx2batch,
                                                                        candi_ent_idxs)
        return batch_rel_loss, batch_pred_rel_tags, len(candi_ent_idxs)

    def forward_sample(self,
                       batch_wc_embs,
                       batch_rnn_outputs,
                       batch_pred_ent_tags,
                       Y_rel,
                       X_len):
        batch_size = len(X_len)
        candi_ent_idxs, labels, idx2batch = self.generate_candidate_entity_pair(batch_pred_ent_tags, Y_rel, X_len)
        if len(candi_ent_idxs) == 0:
            loss = convert_float_variable([0], self.use_cuda)
            loss = loss.squeeze(0)
            loss.requires_grad = True
            return loss, [[] for _ in range(batch_size)], 0

        scores = self.get_score(candi_ent_idxs,
                                idx2batch,
                                batch_pred_ent_tags,
                                batch_wc_embs,
                                batch_rnn_outputs,
                                X_len)
        batch_rel_loss, batch_pred_rel_tags = self.get_loss_and_sample(scores,
                                                                       labels,
                                                                       batch_size,
                                                                       idx2batch,
                                                                       candi_ent_idxs)
        return batch_rel_loss, batch_pred_rel_tags, len(candi_ent_idxs)
    def generate_relation_dict(self, y_rel):
        rel_dict = {}
        for b, e, t in y_rel:
            b = tuple(range(b[0], b[-1]))
            e = tuple(range(e[0], e[-1]))
            rel_dict[(b, e)] = t
        return rel_dict

    def get_entity_idx2chunk_type(self, t_entity):
        entity_idx2chunk_type = {}
        for k, v in t_entity.items():
            for e in v:
                entity_idx2chunk_type[e] = self.ent_model.chunk2idx[k]
        return entity_idx2chunk_type

    def generate_candidate_entity_pair_with_win(self, entity_idx2chunk_type):
        instance_candidate_set = set()
        for ent1_idx in entity_idx2chunk_type.keys():
            for ent2_idx in entity_idx2chunk_type.keys():
                if ent1_idx[0] >= ent2_idx[0]:
                    continue
                if self.win is not None and ent2_idx[0] - ent1_idx[-1] > self.win:
                    continue
                instance_candidate_set.add((ent1_idx, ent2_idx))
        return instance_candidate_set

    def add_gold_candidate(self, instance_candidate_set, y_rel):
        for b, e, t in y_rel:
            b = tuple(range(b[0], b[-1]))
            e = tuple(range(e[0], e[-1]))
            if set(b) & set(e) == set():
                instance_candidate_set.add((b, e))

    def adjust_negative_ratio(self, instance_candidate_set, rel_dict, r=1.0):
        position_num = len(rel_dict)
        negative_num = len(instance_candidate_set) - position_num

        if negative_num <= r * position_num:
            return instance_candidate_set

        negative_num = int(r * position_num)
        negative_instance_list = []
        positive_instance_list = []
        for b, e in instance_candidate_set:
            if (b, e) in rel_dict:
                positive_instance_list.append((b, e))
            else:
                negative_instance_list.append((b, e))
        np.random.shuffle(negative_instance_list)
        negative_instance_list = negative_instance_list[:negative_num]
        return set(positive_instance_list) | set(negative_instance_list)

    def generate_candidate_entity_pair(self, Y, Y_rel, X_len):
        labels = []
        idx2batch = {}
        entity_pair_idxs = []
        for batch_idx in range(len(Y)):
            cur_len = X_len[batch_idx]

            rel_dict = self.generate_relation_dict(Y_rel[batch_idx])
            y = Y[batch_idx][:cur_len]

            y = [self.ent_model.chunk_vocab.idx2item[t] for t in y]
            t_entity = self.get_entity(y)

            entity_idx2chunk_type = self.get_entity_idx2chunk_type(t_entity)

            instance_candidate_set = self.generate_candidate_entity_pair_with_win(entity_idx2chunk_type)

            if self.training and not self.sampling:
                self.add_gold_candidate(instance_candidate_set, Y_rel[batch_idx])
                #  instance_candidate_set = self.adjust_negative_ratio(instance_candidate_set, rel_dict)

            for b, e in instance_candidate_set:
                if (b, e) in rel_dict:
                    t = rel_dict[(b, e)]
                else:
                    t = self.N_ID
                idx2batch[len(entity_pair_idxs)] = batch_idx
                entity_pair_idxs.append((b, e))
                labels.append(t)
        return entity_pair_idxs, labels, idx2batch

    def generate_word_representation(self, Y, X_len, batch_wc_embs, batch_rnn_outputs):
        Z = []
        for batch_idx in range(len(Y)):
            x_len = X_len[batch_idx]
            y = Y[batch_idx][: x_len] # seq_size x 1

            y = convert_long_variable(y, self.use_cuda)
            #  y_embs = self.ent_model.tag_embeddings(y)
            y_embs = self.ent_emb_oh(y)

            rnn_outputs = batch_rnn_outputs[batch_idx][: x_len]
            wc_embs = batch_wc_embs[batch_idx][: x_len]

            #  Z.append(torch.cat([wc_embs, rnn_outputs, y_embs], 1))
            Z.append(torch.cat([rnn_outputs, y_embs], 1))
        return Z

    def get_score(self,
                  candi_ent_idxs,
                  idx2batch,
                  Y,
                  batch_wc_embs,
                  batch_rnn_outputs,
                  X_len):
        Z = self.generate_word_representation(Y,
                                              X_len,
                                              batch_wc_embs,
                                              batch_rnn_outputs)

        final_vecs = self.get_final_vecs(candi_ent_idxs,
                                         idx2batch,
                                         Z,
                                         batch_rnn_outputs,
                                         X_len)

        scores = self.output_seq(final_vecs)
        return scores

    def get_conv_feature(self, candi_ent_idxs, idx2batch, Z, batch_rnn_outputs):
        b_vecs = []
        mid_vecs = []
        e_vecs = []
        for i in range(len(candi_ent_idxs)):
            batch_idx = idx2batch[i]
            z = Z[batch_idx]
            rnn_outputs = batch_rnn_outputs[batch_idx]
            b, e = candi_ent_idxs[i]

            assert b[0] < e[0]

            if b[-1] + 1 == e[0]:
                mid_vecs.append([])
            else:
                #  mid_vecs.append(list(rnn_outputs[b[-1]+1: e[0]].split(1)))
                mid_vecs.append(list(z[b[-1]+1: e[0]].split(1)))

            b_vecs.append(list(z[b[0]: b[-1]+1].split(1)))
            e_vecs.append(list(z[e[0]: e[-1]+1].split(1)))

        #  mid_vecs = self.pad_feature_with_tag(mid_vecs)
        mid_vecs = self.pad_feature(mid_vecs)
        b_vecs = self.pad_feature(b_vecs)
        e_vecs = self.pad_feature(e_vecs)

        mid_vecs = self.get_conv(mid_vecs, self.mid_convs)
        b_vecs = self.get_conv(b_vecs, self.b_convs)
        e_vecs = self.get_conv(e_vecs, self.e_convs)
        return b_vecs, mid_vecs, e_vecs

    def get_distance_between_entity(self, candi_ent_idxs):
        dist_vecs = []
        for i in range(len(candi_ent_idxs)):
            b, e = candi_ent_idxs[i]
            assert b[0] < e[0]
            distance = np.eye(self.max_sent_len)[e[0] - b[-1]]
            distance = convert_float_variable(distance, self.use_cuda).unsqueeze(0)
            dist_vecs.append(distance)
        dist_vecs = torch.cat(dist_vecs, 0)
        return dist_vecs

    def get_forward_segment(self, fward_rnn_output, b, e):
        max_len, h_size = fward_rnn_output.size()
        if b > e:
            zero_vec = convert_float_variable(torch.zeros(h_size), self.use_cuda)
            return zero_vec
        if b == 0:
            return fward_rnn_output[e]
        return fward_rnn_output[e] - fward_rnn_output[b - 1]

    def get_backward_segment(self, bward_rnn_output, b, e):
        max_len, h_size = bward_rnn_output.size()
        if b > e:
            zero_vec = convert_float_variable(torch.zeros(h_size), self.use_cuda)
            return zero_vec
        if e == max_len - 1:
            return bward_rnn_output[b]
        return bward_rnn_output[b] - bward_rnn_output[e + 1]

    def get_segment_feature(self,
                            candi_ent_idxs,
                            batch_rnn_outputs,
                            X_len,
                            idx2batch):
        left_vecs = []
        right_vecs = []
        hidden_size = self.ent_model.hidden_size
        for i in range(len(candi_ent_idxs)):
            b, e = candi_ent_idxs[i]
            batch_idx = idx2batch[i]
            cur_len = X_len[batch_idx]
            rnn_outputs = batch_rnn_outputs[batch_idx]

            fward_rnn_output, bward_rnn_output = rnn_outputs.split(hidden_size // 2, 1)

            fward_left_vec = self.get_forward_segment(fward_rnn_output, 0, b[0] - 1)
            bward_left_vec = self.get_backward_segment(bward_rnn_output, 0, b[0] - 1)
            left_vec = torch.cat([fward_left_vec, bward_left_vec], 0).unsqueeze(0)
            left_vecs.append(left_vec)

            fward_right_vec = self.get_forward_segment(fward_rnn_output, e[-1] + 1, cur_len - 1)
            bward_right_vec = self.get_forward_segment(bward_rnn_output, e[-1] + 1, cur_len - 1)
            right_vec = torch.cat([fward_right_vec, bward_right_vec], 0).unsqueeze(0)
            right_vecs.append(left_vec)
        left_vecs = torch.cat(left_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)
        return left_vecs, right_vecs

    def get_final_vecs(self,
                       candi_ent_idxs,
                       idx2batch,
                       Z,
                       batch_rnn_outputs,
                       X_len):
        b_vecs, mid_vecs, e_vecs = self.get_conv_feature(candi_ent_idxs,
                                                         idx2batch,
                                                         Z,
                                                         batch_rnn_outputs)
        dist_vecs = self.get_distance_between_entity(candi_ent_idxs)

        left_vecs, right_vecs = self.get_segment_feature(candi_ent_idxs,
                                                         batch_rnn_outputs,
                                                         X_len,
                                                         idx2batch)
        #  final_vecs = [relative_pos_vecs, mid_vecs, b_vecs, e_vecs, dist_vecs]
        final_vecs = [mid_vecs, b_vecs, e_vecs, dist_vecs, left_vecs, right_vecs]
        #  final_vecs = [mid_vecs, b_vecs, e_vecs, dist_vecs]
        final_vecs = torch.cat(final_vecs, 1)
        return final_vecs

    def get_conv(self, h, convs):
        h = self.dropout(h)
        h = h.unsqueeze(1) # batch_size x 1 x seq_size x conv_input_size
        h = [F.relu(conv(h)).squeeze(3) for conv in convs] #[(N,Co,W), ...]*len(Ks)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h] #[(N,Co), ...]*len(Ks)
        h = torch.cat(h, 1)
        return h

    def pad_feature(self, features):
        max_len = max([len(e) for e in features])

        def init_pad_h():
            pad_h = Variable(torch.zeros(1, self.conv_input_size))
            if self.use_cuda:
                pad_h = pad_h.cuda(async=True)
            return pad_h

        if max_len == 0:
            return torch.cat([init_pad_h() for _ in features], 0).unsqueeze(1)
        f = []
        for feature in features:
            feature = feature + [init_pad_h() for e in range(max_len - len(feature))]
            feature = torch.cat(feature, 0) # seq_size x conv_input_size
            f.append(feature.unsqueeze(0)) # 1 x seq_size x conv_input_size
        return torch.cat(f, 0) # batch_size x seq_size x conv_input_size

    def pad_feature_with_tag(self, features):
        max_len = max([len(e) for e in features])

        def init_pad_h():
            pad_h = Variable(torch.zeros(1, self.conv_input_size - self.ent_model.tag_size))
            if self.use_cuda:
                pad_h = pad_h.cuda(async=True)
            return pad_h

        if max_len == 0:
            return torch.cat([init_pad_h() for _ in features], 0).unsqueeze(1)
        f = []
        for feature in features:
            feature = feature + [init_pad_h() for e in range(max_len - len(feature))]
            feature = torch.cat(feature, 0) # seq_size x conv_input_size
            f.append(feature.unsqueeze(0)) # 1 x seq_size x conv_input_size
        return torch.cat(f, 0) # batch_size x seq_size x conv_input_size
    def get_loss_and_predict(self,
                             scores,
                             labels,
                             batch_size,
                             idx2batch,
                             candi_ent_idxs):
        labels = convert_long_variable(labels, self.use_cuda)
        _, max_i = scores.max(1)
        if self.use_cuda:
            max_i = max_i.cpu()
        batch_rel_loss = self.loss_function(scores, labels)

        pred_rel_tags = [[] for _ in range(batch_size)]
        for i, (b, e) in enumerate(candi_ent_idxs):
            cur_i = max_i[i].data.numpy()
            if cur_i != self.N_ID:
                pred_rel_tags[idx2batch[i]].append((b, e, cur_i))
        return batch_rel_loss, pred_rel_tags

    def get_loss_and_sample(self,
                            scores,
                            labels,
                            batch_size,
                            idx2batch,
                            candi_ent_idxs):
        labels = convert_long_variable(labels, self.use_cuda)
        _, max_i = scores.max(1)
        if self.use_cuda:
            max_i = max_i.cpu()
        probs = self.Softmax(scores)
        sample_y = Variable(probs.multinomial(1).data.squeeze(1))
        batch_sample_rel_loss = self.loss_function(scores, sample_y)
        sample_rel_tags = [[] for _ in range(batch_size)]
        for i, (b, e) in enumerate(candi_ent_idxs):
            cur_i = sample_y[i].cpu().data.numpy()
            if cur_i != self.N_ID:
                sample_rel_tags[idx2batch[i]].append((b, e, cur_i))
        return batch_sample_rel_loss, sample_rel_tags

    def get_entity(self, y):
        last_guessed = 'O'        # previously identified chunk tag
        last_guessed_type = ''    # type of previous chunk tag in corpus
        guessed_idx = []
        t_guessed_entity2idx = defaultdict(list)
        for i, tag in enumerate(y):
            guessed, guessed_type = parse_tag(tag)
            start_guessed = start_of_chunk(last_guessed, guessed,
                                                last_guessed_type, guessed_type)
            end_guessed = end_of_chunk(last_guessed, guessed,
                                            last_guessed_type, guessed_type)
            if start_guessed:
                if guessed_idx:
                    t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
                guessed_idx = [guessed_type, i]
            elif guessed_idx and not start_guessed and guessed_type == guessed_idx[0]:
                guessed_idx.append(i)

            last_guessed = guessed
            last_guessed_type = guessed_type
        if guessed_idx:
            t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
        return t_guessed_entity2idx

class JointEntRelModel(nn.Module):

    def __init__(self,
                 word_char_embedding,
                 word_char_emb_size,
                 out_channels,
                 kernel_sizes,
                 hidden_size,
                 parse_lstm_size,
                 tag_emb_size,
                 position_emb_size,
                 tag_size,
                 relation_size,
                 max_sent_len,
                 chunk_vocab,
                 N_ID,
                 num_layers=1,
                 use_cuda=False,
                 bidirectional=True,
                 win=15,
                 sch_k=0.5,
                 dropout=0.5):
        super(JointEntRelModel, self).__init__()
        self.use_cuda = use_cuda
        self.sch_k = sch_k

        self.entity_model = EntModel(word_char_embedding,
                                     word_char_emb_size,
                                     hidden_size,
                                     parse_lstm_size,
                                     tag_emb_size,
                                     tag_size,
                                     chunk_vocab,
                                     num_layers=num_layers,
                                     use_cuda=use_cuda,
                                     bidirectional=bidirectional,
                                     dropout=dropout)

        self.rel_model = RelModel(self.entity_model,
                                  N_ID,
                                  out_channels,
                                  kernel_sizes,
                                  relation_size,
                                  position_emb_size,
                                  max_sent_len,
                                  use_cuda=use_cuda,
                                  dropout=dropout,
                                  win=win)

    def forward(self, X, X_char, X_lstm_h, X_len, X_mask, Y, Y_rel, i_epoch=0):
        batch_ent_loss, batch_pred_ent_tags, batch_wc_embs, batch_rnn_outputs = self.entity_model(X, X_char, X_lstm_h, X_len, X_mask, Y)
        if self.training:
            sch_sample_ent_tags = self.schedule_sample(batch_pred_ent_tags, Y, i_epoch)
        else:
            sch_sample_ent_tags = batch_pred_ent_tags
        batch_rel_loss, batch_pred_rel_tags, candi_rel_num = self.rel_model(batch_wc_embs,
                                                                            batch_rnn_outputs,
                                                                            sch_sample_ent_tags,
                                                                            Y_rel,
                                                                            X_len)

        return batch_ent_loss, batch_rel_loss, batch_pred_ent_tags, batch_pred_rel_tags, candi_rel_num

    def forward_sample(self, X, X_char, X_lstm_h, X_len, X_mask, sample_Y, Y_rel):
        batch_ent_loss, batch_sample_ent_tags, batch_wc_embs, batch_rnn_outputs = self.entity_model(X, X_char, X_lstm_h, X_len, X_mask, sample_Y)
        sample_Y = sample_Y.cpu().numpy()
        batch_sample_rel_loss, batch_sample_rel_tags, candi_rel_num = self.rel_model.forward_sample(batch_wc_embs,
                                                                                                    batch_rnn_outputs,
                                                                                                    sample_Y,
                                                                                                    Y_rel,
                                                                                                    X_len)
        return batch_ent_loss, batch_sample_rel_loss, sample_Y, batch_sample_rel_tags, candi_rel_num

    def schedule_sample(self, pred_tags, Y, i_epoch):
        sch_p = self.sch_k / (self.sch_k + np.exp(i_epoch / self.sch_k))
        sch_tags = []
        for i, tags in enumerate(pred_tags):
            each_tags = []
            for j, tag in enumerate(tags):
                rd = np.random.random()
                if rd <= sch_p:
                    each_tags.append(Y[i][j])
                else:
                    each_tags.append(tag)
            sch_tags.append(each_tags)
        return sch_tags
