#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/06 17:00:26

@author: Changzhi Sun
"""

class Vocab(object):

    def __init__(self,
                 name="vocab",
                 offset_items=tuple([]),
                 PAD=None,
                 lower=True):
        self.name = name
        self.item2idx = {}
        self.idx2item = []
        self.size = 0
        self.PAD = PAD
        self.lower=lower
        
        self.batch_add(offset_items, lower=False)
        if PAD is not None:
            self.add(PAD, lower=False)
            self.PAD_ID = self.item2idx[self.PAD]
        self.offset = self.size
        
    def add(self, item, lower=True):
        if self.lower and lower:
            item = item.lower()
        if item not in self.item2idx:
            self.item2idx[item] = self.size
            self.size += 1
            self.idx2item.append(item)
            
    def batch_add(self, items, lower=True):
        for item in items:
            self.add(item, lower=lower)
            
    def in_vocab(self, item, lower=True):
        if self.lower and lower:
            item = item.lower()
        return item in self.item2idx
        
    def getidx(self, item, lower=True):
        if self.lower and lower:
            item = item.lower()
        if item not in self.item2idx:
            if self.PAD is None:
                raise RuntimeError("PAD is not defined. %s not in vocab." % item)
            return self.PAD_ID
        return self.item2idx[item]
            
    def __repr__(self):
        return "Vocab(name={}, size={:d}, PAD={}, offset={:d}, lower={})".format(
            self.name, self.size,
            self.PAD, self.offset,
            self.lower
        )
