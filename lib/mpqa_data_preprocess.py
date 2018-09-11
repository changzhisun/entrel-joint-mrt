#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/04 20:35:21

@author: Changzhi Sun
"""
import os
import numpy as np
import json
from itertools import chain

def iter_sents(annot_path):
    with open(annot_path, "r", encoding="utf8") as f:
        sent = {'token': [], 'pos': [], 'tag': []}
        for line in f:
            if line.strip() == "":
                yield sent
                sent = {'token': [], 'pos': [], 'tag': []}
            else:
                token, pos, tag = line.strip().split('\t')
                sent['token'].append(token)
                sent['pos'].append(pos)
                sent['tag'].append(tag)


def iter_sents_id(sentenceid_path):
    with open(sentenceid_path, "r", encoding="utf8") as f:
        for line in f:
            yield line.strip()


def get_opinions(sent_tag, opin_str):
    ret_indices = []
    ret_tag = []
    i = 0
    while i < len(sent_tag):
        if sent_tag[i].startswith("B_" + opin_str):
            ret_tag.append(sent_tag[i])
            indices = [i]
            i += 1
            while i < len(sent_tag) and sent_tag[i].startswith(opin_str):
                indices.append(i)
                i += 1
            ret_indices.append(indices)
        else:
            i += 1
    return ret_indices, ret_tag


def get_opin_expr(sent_tag, expr_str="DSE"):
    return get_opinions(sent_tag, expr_str)


def get_opin_target(sent_tag, target_str="TARGET"):
    return get_opinions(sent_tag, target_str)


def get_opin_holder(sent_tag, holder_str="AGENT"):
    return get_opinions(sent_tag, holder_str)


def has_overlap(opin_list):
    for e1 in opin_list:
        for e2 in opin_list:
            if e1 == e2:
                continue
            if set(e1) & set(e2):
                print(e1, e2)


def add_element(rel2tri, opin_tag):
    for i, tag in enumerate(opin_tag):
        tag_split = tag.split('_')
        if len(tag_split)< 3:
            #  print(tag)
            continue
        for e in tag_split[2:]:
            if e not in rel2tri:
                # (T, O, H)
                rel2tri[e] = [-1, -1, -1]
            if tag_split[1] == "TARGET":
                rel2tri[e][0] = i
            elif tag_split[1] == "DSE":
                rel2tri[e][1] = i
            elif tag_split[1] == "AGENT":
                rel2tri[e][2] = i


def get_relation(rel2tri, idx1, idx2):
    ret_rel = []
    for key, value in rel2tri.items():
        if value[idx1] != -1 and value[idx2] != -1:
            ret_rel.append([value[idx1], value[idx2]])
    return ret_rel

def get_impl_relation(rel2tri, O_idx, idx1, idx2):
    ret_rel = []
    for key, value in rel2tri.items():
        if value[O_idx] != -1 and value[idx1] != -1 and value[idx2] == -1:
            ret_rel.append(value)
    return ret_rel

def get_only_relation(rel2tri, idx, idx1, idx2):
    ret_rel = []
    for key, value in rel2tri.items():
        if value[idx] != -1 and value[idx1] == -1 and value[idx2] == -1:
            ret_rel.append(value)
    return ret_rel

def generate_cv_data(mpqa_dir, cv_index, sents):
    if not os.path.exists(os.path.join(MPQA_dir, "cv%d" % cv_index)):
        os.mkdir("cv%d" % cv_index)
        os.mkdir(os.path.join(MPQA_dir, "cv%d" % cv_index, "train"))
        os.mkdir(os.path.join(MPQA_dir, "cv%d" % cv_index, "dev"))
        os.mkdir(os.path.join(MPQA_dir, "cv%d" % cv_index, "test"))
    train_docs = set(iter_sents_id(os.path.join(mpqa_dir, "filelist_train%d" % cv_index)))
    test_docs = set(iter_sents_id(os.path.join(mpqa_dir, "filelist_test%d" % cv_index)))
    f1 = open(os.path.join(MPQA_dir, "cv%d" % cv_index, "train", "data.json"), "w")
    f2 = open(os.path.join(MPQA_dir, "cv%d" % cv_index, "test", "data.json"), "w")
    f3 = open(os.path.join(MPQA_dir, "cv%d" % cv_index, "dev", "data.json"), "w")
    for sent in sents:
        if sent['articleId'] in train_docs:
            print(json.dumps(sent), file=f1)
        elif sent['articleId'] in test_docs:
            print(json.dumps(sent), file=f2)
        else:
            print(json.dumps(sent), file=f3)
    f1.close()
    f2.close()
    f3.close()


if __name__ == "__main__":

    processed_dataset_dir = "../data/MPQA/mpqa"
    MPQA_dir = "../data/MPQA"
    sentenceid_file = "sentenceid.txt"
    annot_file = "all_ILP.txt"

    sents = []
    for sent_id, sent in zip(iter_sents_id(os.path.join(processed_dataset_dir, sentenceid_file)),
                             iter_sents(os.path.join(processed_dataset_dir, annot_file))):
        new_sent = {}
        article_id, sent_id, article_str = sent_id.split(' ')
        new_sent['articleId'] = article_str
        new_sent['sentId'] = sent_id
        new_sent['sentText'] = " ".join(sent['token'])
        new_sent['sentPOS'] = " ".join(sent['pos'])
        new_sent['entityMentions'] = []
        new_sent['relationMentions'] = []

        opin_expr, opin_expr_tag = get_opin_expr(sent['tag'], "DSE")
        for ent_indice, ent_id in zip(opin_expr, opin_expr_tag):
            em_mention = {}
            em_mention['emId'] = ent_id
            em_mention['offset'] = [ent_indice[0], ent_indice[-1] + 1]
            em_mention['text'] = " ".join([sent['token'][e] for e in ent_indice])
            em_mention['label'] = "OpinExpr"
            new_sent['entityMentions'].append(em_mention)

        opin_target, opin_target_tag = get_opin_target(sent['tag'], "TARGET")
        for ent_indice, ent_id in zip(opin_target, opin_target_tag):
            em_mention = {}
            em_mention['emId'] = ent_id
            em_mention['offset'] = [ent_indice[0], ent_indice[-1] + 1]
            em_mention['text'] = " ".join([sent['token'][e] for e in ent_indice])
            em_mention['label'] = "OpinTarget"
            new_sent['entityMentions'].append(em_mention)

        opin_holder, opin_holder_tag = get_opin_holder(sent['tag'], "AGENT")
        for ent_indice, ent_id in zip(opin_holder, opin_holder_tag):
            em_mention = {}
            em_mention['emId'] = ent_id
            em_mention['offset'] = [ent_indice[0], ent_indice[-1] + 1]
            em_mention['text'] = " ".join([sent['token'][e] for e in ent_indice])
            em_mention['label'] = "OpinHolder"
            new_sent['entityMentions'].append(em_mention)

        rel2tri = {}
        add_element(rel2tri, opin_expr_tag)
        add_element(rel2tri, opin_target_tag)
        add_element(rel2tri, opin_holder_tag)

        #  has_overlap(opin_expr)
        #  has_overlap(opin_target)
        #  has_overlap(opin_holder)
        #  has_overlap(opin_expr + opin_target + opin_holder)

        isabout = get_relation(rel2tri, 0, 1)
        for e in isabout:
            opin_str1 = " ".join([sent['token'][k] for k in opin_target[e[0]]])
            opin_str2 = " ".join([sent['token'][k] for k in opin_expr[e[1]]])

            rel_mention = {}
            rel_mention['label'] = 'IS-ABOUT'
            rel_mention['em1Id'] = opin_target_tag[e[0]]
            rel_mention['em2Id'] = opin_expr_tag[e[1]]
            rel_mention['em1Text'] = opin_str1
            rel_mention['em2Text'] = opin_str2

            new_sent['relationMentions'].append(rel_mention)

        isfrom = get_relation(rel2tri, 1, 2)
        for e in isfrom:
            opin_str1 = " ".join([sent['token'][k] for k in opin_expr[e[0]]])
            opin_str2 = " ".join([sent['token'][k] for k in opin_holder[e[1]]])

            rel_mention = {}
            rel_mention['label'] = 'IS-FROM'
            rel_mention['em1Id'] = opin_expr_tag[e[0]]
            rel_mention['em2Id'] = opin_holder_tag[e[1]]
            rel_mention['em1Text'] = opin_str1
            rel_mention['em2Text'] = opin_str2

            new_sent['relationMentions'].append(rel_mention)

        #  isabout_implicit = get_impl_relation(rel2tri, 1, 2, 0)
        #  isfrom_implicit = get_impl_relation(rel2tri, 1, 0, 2)
        #  expr_only = get_only_relation(rel2tri, 1, 0, 2)
        #  target_only = get_only_relation(rel2tri, 0, 1, 2)
        #  holder_only = get_only_relation(rel2tri, 2, 0, 1)

        #  print("###")
        #  for key, value in rel2tri.items():
            #  print(value)
        sents.append(new_sent)
    print("sentence num:", len(sents))
    for i in range(10):
        generate_cv_data(os.path.join(processed_dataset_dir, "data_MPQA"), i, sents)

