#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/03 17:36:11

@author: Changzhi Sun
"""
import numpy as np
import json
import os
import re

def read_line(filename):
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            yield line.strip()


def build_offset_mapping(text):
    ch_idx = 0
    mapping= {}
    for sent_id, sent in enumerate(text):
        tok_idx = 0
        for ch in sent:
            mapping[ch_idx] = (sent_id, tok_idx)
            if ch == ' ':
                tok_idx += 1
            ch_idx += 1
        mapping[ch_idx] = (sent_id, tok_idx)
        ch_idx += 1
    return mapping


def parse_file(filename_prefix):
    sents = []
    pattern = r"^\d{0,4}-\d{0,2}-\d{0,2}\s\d{0,2}:\d{0,2}:\d{0,2}$"
    ftext = open(filename_prefix + ".split.txt", "r", encoding="utf8")
    fann= open(filename_prefix + ".split.ann", "r", encoding="utf8")
    text = ""
    for line in ftext:
        text += line
    sent_text = text.split('\n')
    if sent_text[-1] == "":
        sent_text.pop()
    if re.search(pattern, sent_text[-1]):
        sent_text.pop()
    char2sent_token = build_offset_mapping(sent_text)
    sents = [e.split(' ') for e in sent_text]

    ann_sents = []
    articleId = filename_prefix.split('/')[-1]
    for i in range(3, len(sents)):
        ann_sent = {'sentId': i - 3,
                    'sentText': sent_text[i],
                    'articleId': articleId,
                    'entityMentions': [],
                    'relationMentions': []}
        ann_sents.append(ann_sent)

    emId2sent_text = {}
    for line in fann:
        if line[0] == 'T':
            type_str, ent_off, ent_str = line.strip().split('\t')
            ent_type, start, end = ent_off.split(' ')
            start, end = int(start), int(end)

            assert text[start:end] == ent_str

            sent_id, tok1_id = char2sent_token[start]
            sent_id_tmp, tok2_id = char2sent_token[end]

            assert sent_id == sent_id_tmp

            restruc_str = " ".join([sents[sent_id][e] for e in range(tok1_id, tok2_id+1)])
            assert restruc_str == ent_str

            idx = type_str.index('-')
            emId = type_str[idx+1:]

            ent_mention = {}
            ent_mention['emId'] = emId
            ent_mention['text'] = ent_str
            ent_mention['offset'] = [tok1_id, tok2_id + 1]
            ent_mention['label'] = ent_type

            ann_sents[sent_id - 3]['entityMentions'].append(ent_mention)
            emId2sent_text[emId] = [sent_id - 3, ent_str, tok1_id]

        elif line[0] == 'R':
            rel_str = line.strip().split('\t')[1]
            rel_type, em1, em2 = rel_str.split(' ')
            em1 = em1[em1.index('-') + 1:]
            em2 = em2[em2.index('-') + 1:]

            assert em1 in emId2sent_text
            assert em2 in emId2sent_text

            sent_id, em1_text, em1_id = emId2sent_text[em1]
            sent_id_tmp, em2_text, em2_id = emId2sent_text[em2]
            if rel_type == "PER-SOC" and em1_id > em2_id:
                em1, em2 = em2, em1
                em1_text, em2_text = em2_text, em1_text
            assert sent_id == sent_id_tmp

            rel_mention = {}
            rel_mention['em1Id'] = em1
            rel_mention['em2Id'] = em2
            rel_mention['em1Text'] = em1_text
            rel_mention['em2Text'] = em2_text
            rel_mention['label'] = rel_type

            ann_sents[sent_id]['relationMentions'].append(rel_mention)
        else:
            print("Error")
    ftext.close()
    fann.close()
    return ann_sents

def process_data_and_save_json(file_gen, save_dir):
    i = 0
    f = open(os.path.join(save_dir, "data.json"), "w", encoding="utf8")
    rel_ct = 0
    rel_set = set()
    ent_set = set()
    for each_file in file_gen:
        sents = parse_file(os.path.join(ace2005_dir, "ace2005", "text", each_file))
        for sent in sents:
            print(json.dumps(sent), file=f)
            rel_ct += len(sent['relationMentions'])
            for e in sent['relationMentions']:
                rel_set.add(e['label'])
            for e in sent['entityMentions']:
                ent_set.add(e['label'])
        i += 1
    print("===================================================================")
    print("%s set Relation Num " % save_dir, rel_ct)
    print(list(rel_set))
    print(list(ent_set))
    print()
    f.close()

ace2005_dir = "../data/ACE2005"

train_gen = read_line(os.path.join(ace2005_dir, "ace2005/split/split_train"))
dev_gen = read_line(os.path.join(ace2005_dir, "ace2005/split/split_dev"))
test_gen = read_line(os.path.join(ace2005_dir, "ace2005/split/split_test"))
process_data_and_save_json(train_gen, os.path.join(ace2005_dir, "train"))
process_data_and_save_json(dev_gen, os.path.join(ace2005_dir, "dev"))
process_data_and_save_json(test_gen, os.path.join(ace2005_dir, "test"))
