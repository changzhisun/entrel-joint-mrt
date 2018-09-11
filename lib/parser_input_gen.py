#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/10/27 09:53:55

@author: Changzhi Sun
"""
import argparse
import os
import json

import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = utils.read_config(config_file_path)

train_file, dev_file, test_file = "train/data.json", "dev/data.json", "test/data.json"
TRAIN_CORPUS = os.path.join("../data/", config['data']['corpus'], train_file)
DEV_CORPUS = os.path.join("../data/", config['data']['corpus'], dev_file)
TEST_CORPUS = os.path.join("../data/", config['data']['corpus'], test_file)

TRAIN_SAVE = os.path.join("../data/", config['data']['corpus'], "train/train.conll")
DEV_SAVE = os.path.join("../data/", config['data']['corpus'], "dev/dev.conll")
TEST_SAVE = os.path.join("../data/", config['data']['corpus'], "test/test.conll")

def conll_input_gen(input_file, save_file):
    fsave = open(save_file, "w", encoding="utf8")
    with open(input_file, "r", encoding="utf8") as f:
        for line in f:
            sent = json.loads(line)
            for i, w in enumerate(sent['sentText'].split(' '), 1):
                print("{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_".format(i, w), file=fsave)
            print(file=fsave)
    fsave.close()

conll_input_gen(TRAIN_CORPUS, TRAIN_SAVE)
conll_input_gen(DEV_CORPUS, DEV_SAVE)
conll_input_gen(TEST_CORPUS, TEST_SAVE)
