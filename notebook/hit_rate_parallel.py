from multiprocessing import Process, Manager, cpu_count, Pool, current_process
import nltk
from collections import defaultdict, Counter
import gensim, logging
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
from gensim import utils, matutils
import os
import collections
from random import shuffle
from copy import deepcopy
from collections import OrderedDict
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from numpy import exp, dot, zeros, outer, dtype, float32 as REAL
import annoy
import itertools

def extract_patterns_in_words(patterns,word1,word2,max_len):
    i = 1
    while(word1[:i] == word2[:i]):
        i = i + 1
    if i != 1 and i > max(len(word1[i-1:]), len(word2[i-1:])) < max_len:
        if ("suffix",word1[i-1:], word2[i-1:]) in patterns:
            patterns[("suffix",word1[i-1:], word2[i-1:])].append((word1, word2))
        else:
            patterns[("suffix",word1[i-1:], word2[i-1:])] = [(word1, word2)]
#         patterns[("suffix",word1[i-1:], word2[i-1:], word1, word2)] += 1
    i = 1
    while(word1[-i:] == word2[-i:]):
        i = i + 1
    if i != 1 and max(len(word1[:-i+1]), len(word2[:-i+1])) < max_len:
        if ("prefix",word1[:-i+1], word2[:-i+1]) in patterns:
            patterns[("prefix",word1[:-i+1], word2[:-i+1])].append((word1, word2))
        else:
            patterns[("prefix",word1[:-i+1], word2[:-i+1])] = [(word1, word2)]
#         patterns[("prefix",word1[:-i+1], word2[:-i+1], word1, word2)] += 1
    return patterns


def build_pattern_dict(vocab,max_len = 6):
    if os.path.exists('../data/patterns_'+ str(len(vocab))):
        patterns_file_r = open('../data/patterns_'+ str(len(vocab)), 'rb')
        patterns = pickle.load(patterns_file_r)
    else:
        patterns  = defaultdict(list)
        for word in vocab:
            for second_word in vocab:
                if word != second_word:
                    extract_patterns_in_words(patterns,word,second_word,max_len)
        patterns_file_w = open('../data/patterns_'+ str(len(vocab)),"wb" )
        pickle.dump(patterns, patterns_file_w)
        patterns_file_w.close()
    return patterns


def downsample_patterns():
    #Downsample to include only top 1000
    pattern_1000 = defaultdict(list)
    for pattern,items in patterns.items():
        shuffle(items)
        pattern_1000[pattern] = items[:1000]
    return pattern_1000

def pair_wise_similarity(word_pair1, word_pair2,annoy_index=None, topn = 10):
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn)
#     print (word_pair2[1])
#     print (closest_n)
    for word, cos_sim in closest_n:
        if word == word_pair2[1]:
            return True
    return False

def annoy_pair_wise_similarity(word_pair1, word_pair2,annoy_index, topn = 10):
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn, indexer=annoy_index)
#     print (word_pair2[1])
#     print (closest_n)
    for word, cos_sim in closest_n:
        if word == word_pair2[1]:
            return True
    return False

def get_similarity_rank(word_pair1, word_pair2, topn=500):
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn)
#     print (word_pair2[1])
#     print (closest_n)
    for n,(word, cos_sim) in enumerate(closest_n):
        if word == word_pair2[1]:
            return (n, cos_sim)
    return (None, None)


def get_hit_rate(patterns, similarity_function, annoy_index=None):
    if False:
        hit_rate_file_r = open('../data/hitrate_'+ str(len(word_vectors.vocab)), 'rb')
        hit_rates_rules = pickle.load(hit_rate_file_r)
        return hit_rates_rules
    else:
        hit_rate_file_w = open('../data/hitrate_'+ str(len(word_vectors.vocab)),"wb" )
        hit_rates_rules = {}
        for (pattern,support_set) in patterns.items():
            hit_rates_word_pair = {}
            for pair1 in support_set:
                hit_count = 0
                hit_pairs = set()
                for pair2 in support_set:
                    if pair1 != pair2 and similarity_function(pair1, pair2, annoy_index,10):
                        hit_count += 1
                        hit_pairs.add(pair2)
                if len(support_set) ==1:
                    total = 1
                else:
                    total = len(support_set) - 1
                if hit_count != 0:
                    hit_rates_word_pair[pair1] =  hit_pairs
            if len(support_set) != 1 and hit_rates_word_pair:
                hit_rates_rules[pattern] = hit_rates_word_pair
        pickle.dump(hit_rates_rules, hit_rate_file_w)
        hit_rate_file_w.close()
        return hit_rates_rules
    

def get_annoy():
    annoy_file_name = '../data/annoy_index__100w2v_3000000'
    if os.path.exists(annoy_file_name):
        annoy_index = AnnoyIndexer()
        annoy_index.load(annoy_file_name)
#         annoy_index.model = word_vectors
    else:
        annoy_index = AnnoyIndexer(word_vectors,dims)
        annoy_index.save(annoy_file_name)
    return annoy_index


hit_rates_rules = {}
def get_hit_rules(t):
    pattern, support_set = t
    hit_rates_word_pair = {}
    for pair1 in support_set:
        hit_count = 0
        hit_pairs = set()
        for pair2 in support_set:
            if pair1 != pair2 and annoy_pair_wise_similarity(pair1, pair2, annoy_index,10):
                hit_count += 1
                hit_pairs.add(pair2)
        if hit_count:
            hit_rates_word_pair[pair1] =  hit_pairs
    if len(support_set) != 1 and hit_rates_word_pair:
        return pattern, hit_rates_word_pair
    return pattern, hit_rates_word_pair

def iterator_slice(iterator, length):
    iterator = iter(iterator)
    while True:
        res = tuple(itertools.islice(iterator, length))
        if not res:
            break
        yield res
 
# word_vectors = KeyedVectors.load_word2vec_format('/home/raja/models/GoogleNews-vectors-negative300.bin.gz', binary=True)
if __name__ == '__main__':
    word_vectors = KeyedVectors.load_word2vec_format('/home/raja/models/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=2000)

    patterns = build_pattern_dict(word_vectors.vocab.keys())
    print ("done with building")

    sampled_patterns = downsample_patterns()
    sampled_patterns_w = open('../data/sampled_patterns_'+ str(len(word_vectors.vocab)),"wb" )
    pickle.dump(sampled_patterns,sampled_patterns_w)
    sampled_patterns_w.close()

    annoy_index = get_annoy()
    

    #hit_rates = Manager().dict()
    hit_rates = {}
        
    pool = Pool()
    works = ((pattern, support_set,) for pattern,support_set in sampled_patterns.items())
    cursor_iterator = iterator_slice(works, 1)
    #print (pool.apply(get_hit_rules,next(cursor_iterator)))
    t = pool.imap(get_hit_rules,works)
    for pattern, hit_rates_word_pair in t:
        hit_rates[pattern] = hit_rates_word_pair

    #hit_rates[pattern] = hit_rates_word_pair
#     print (collections.Counter(pids))
    print (len(hit_rates))


#     hit_rates = get_hit_rate(sampled_patterns, annoy_pair_wise_similarity, annoy_index)
    output_hit_rates = {}
    output_hit_rates.update(hit_rates)
    hit_rate_file_w = open('../data/hitrate_'+ str(len(word_vectors.vocab)),"wb" )
    pickle.dump(hit_rates, hit_rate_file_w)
    hit_rate_file_w.close()
