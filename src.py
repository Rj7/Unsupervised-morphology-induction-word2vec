import nltk
from collections import defaultdict, Counter
import gensim
import logging
import nltk
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
from multiprocessing import Process, Pool
import os
import collections
import datetime

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename='morph_induction.log')

def extract_patterns_in_words(patterns,pattern_counter,word1,word2,max_len):
    i = 1
    while(word1[:i] == word2[:i]):
        i = i + 1
    if i != 1 and i > max(len(word1[i-1:]), len(word2[i-1:])) < max_len:
        pattern_counter[("suffix",word1[i-1:], word2[i-1:])] += 1
        if ("suffix",word1[i-1:], word2[i-1:]) in patterns:
            patterns[("suffix",word1[i-1:], word2[i-1:])].append((word1, word2))
        else:
            patterns[("suffix",word1[i-1:], word2[i-1:])] = [(word1, word2)]
#         patterns[("suffix",word1[i-1:], word2[i-1:], word1, word2)] += 1
    i = 1
    while(word1[-i:] == word2[-i:]):
        i = i + 1
    if i != 1 and max(len(word1[:-i+1]), len(word2[:-i+1])) < max_len:
        pattern_counter[("prefix",word1[:-i+1], word2[:-i+1])] += 1
        if ("prefix",word1[:-i+1], word2[:-i+1]) in patterns:
            patterns[("prefix",word1[:-i+1], word2[:-i+1])].append((word1, word2))
        else:
            patterns[("prefix",word1[:-i+1], word2[:-i+1])] = [(word1, word2)]
#         patterns[("prefix",word1[:-i+1], word2[:-i+1], word1, word2)] += 1
    return patterns


def build_pattern_dict(vocab,max_len = 6):
    patterns  = defaultdict(int)
    pattern_counter = Counter()
    for word in vocab:
        for second_word in vocab:
            if word != second_word:
                extract_patterns_in_words(patterns,pattern_counter,word,second_word,max_len)
    return patterns, pattern_counter

def index_vector(word_vectors, dimensions=300):
    fname = 'data/annoy.index'
    # Persist index to disk
    if os.path.exists(fname):
        annoy_index = AnnoyIndexer()
        annoy_index.load(fname)
        annoy_index.model = word_vectors
    else:
        annoy_index = AnnoyIndexer(word_vectors, dimensions)
    annoy_index.save(fname)
    return annoy_index


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    word_vectors = KeyedVectors.load_word2vec_format('/home/raja/GoogleNews-vectors-negative300.bin.gz', binary=True)
    logging.info("Length of the Vocab: %s", len(word_vectors.vocab))
    
    logging.info("Normalizing Vectors")
    word_vectors.init_sims()


    logging.info ("Building Annoy Indexing")
    annoy_index = index_vector(word_vectors=word_vectors, dimensions=300)
    
    annoy_end_time = datetime.datetime.now()
    logging.info ("Finished Building Annoy Indexing: %s", annoy_end_time - start_time)



