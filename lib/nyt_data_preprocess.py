#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/03 17:40:42

@author: Changzhi Sun
"""
import json
import numpy as np
import os


def token2offset(tokens):
    tok2off = {}
    for i, tok in enumerate(tokens):
        if tok not in tok2off:
            tok2off[tok] = []
        tok2off[tok].append(i)
    return tok2off


def offset_of_tokens(en_text, tok2off, tokens):
    toks = en_text.split(' ')
    for first_idx in tok2off[toks[0]]:
        flag = True
        for i in range(len(toks)):
            if tokens[first_idx + i] != toks[i]:
                flag = False
                break
        if flag:
            return (first_idx, first_idx + len(toks))


def replace_latin(string):
    r = {'á': 'a',
         'â': 'a',
         'Á': 'A',
         'É': 'E',
         'è': 'e',
         'é': 'e',
         'ê': 'e',
         'í': 'i',
         'ó': 'o',
         'ô': 'o',
         'ö': 'o',
         'Ó': 'O',
         'ú': 'u',
         'ü': 'u',
         'ñ': 'n'}
    for k, v in r.items():
        string = string.replace(k, v)
    return string


def convert_format(sent):
    new_sent = {'sentId': sent['sentId'],
                'articleId': sent['articleId'],
                'sentText': replace_latin(sent['sentText'].strip()),
                'entityMentions': [],
                'relationMentions': []}

    if new_sent['sentText'][0] == '"' and new_sent['sentText'][-1] == '"':
        new_sent['sentText'] = new_sent['sentText'][1:-1]

    tokens = new_sent['sentText'].split(' ')

    tok2off = token2offset(tokens)
    ent2id = {}

    for ent_mention in sent['entityMentions']:
        new_ent_mention = {}
        new_ent_mention['emId'] = ent_mention['start']
        new_ent_mention['label'] = ent_mention['label']
        new_ent_mention['text'] = replace_latin(ent_mention['text'])

        toks = new_ent_mention['text'].split(' ')

        assert toks[0] in tok2off

        new_ent_mention['offset'] = offset_of_tokens(new_ent_mention['text'], tok2off, tokens)

        assert new_ent_mention['offset'] is not None

        recon_txt = " ".join(
            [tokens[e] for e in range(new_ent_mention['offset'][0],
                                      new_ent_mention['offset'][1])])
        assert new_ent_mention['text'] == recon_txt

        ent2id[new_ent_mention['text']] = new_ent_mention['emId']
        new_sent['entityMentions'].append(new_ent_mention)

    none_ct = 0
    for rel_mention in sent['relationMentions']:
        if rel_mention['label'] == "None":
            none_ct += 1
            continue

        new_rel_mention = {}
        new_rel_mention['em1Text'] = replace_latin(rel_mention['em1Text'])
        new_rel_mention['em2Text'] = replace_latin(rel_mention['em2Text'])
        new_rel_mention['label'] = rel_mention['label']

        #  print(new_rel_mention['em2Text'])
        #  print(ent2id)

        new_rel_mention['em1Id'] = ent2id[new_rel_mention['em1Text']]
        new_rel_mention['em2Id'] = ent2id[new_rel_mention['em2Text']]
        new_sent['relationMentions'].append(new_rel_mention)
    return new_sent, len(sent['relationMentions']), none_ct


def process_data_and_save_json(data, save_dir):
    with open(os.path.join(save_dir, "data.json"), 'w') as g:
        rel_all = 0
        rel_none = 0
        for sent in data:
            sent, r_all, r_none = convert_format(sent)
            rel_all += r_all
            rel_none += r_none
            print(json.dumps(sent), file=g)
        print("===================================================================")
        print("%s set Relation Num (include None relation):" % save_dir, rel_all)
        print("%s set None Relation Num:" % save_dir, rel_none)
        print("%s set Relation Num(exclude None relation):" % save_dir, rel_all - rel_none)
        print()


nyt_dir = "../data/NYT"

with open(os.path.join(nyt_dir, "nyt/train.json"), "r") as f:
    data = []
    for line in f:
        sent = json.loads(line)
        data.append(sent)

process_data_and_save_json(data, os.path.join(nyt_dir, "train"))

with open(os.path.join(nyt_dir, "nyt/test.json"), "r") as f:
    data = []
    for line in f:
        data.append(json.loads(line))

np.random.seed(0)
np.random.shuffle(data)
dev_len = int(len(data) * 0.1)
dev_data, test_data = data[:dev_len], data[dev_len:]

process_data_and_save_json(dev_data, os.path.join(nyt_dir, "dev"))
process_data_and_save_json(test_data, os.path.join(nyt_dir, "test"))

