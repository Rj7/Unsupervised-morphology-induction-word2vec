import nltk
from collections import defaultdict, Counter
import gensim, logging
import nltk
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
from multiprocessing import Process, Pool
import os
import datetime
from random import shuffle
from copy import deepcopy
from collections import OrderedDict
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter

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
    if os.path.exists('data/patterns_'+ str(len(vocab))):
        patterns_file_r = open('data/patterns_'+ str(len(vocab)), 'rb')
        pattern_counter_file_r = open('data/patterns_counter_'+ str(len(vocab)), 'rb')
        patterns = pickle.load(patterns_file_r)
        pattern_counter = pickle.load(pattern_counter_file_r)
    else:
        patterns_file_w = open('data/patterns_'+ str(len(vocab)),"wb" )
        pattern_counter_file_w = open('data/patterns_counter_'+ str(len(vocab)),"wb" )
        patterns  = defaultdict(list)
#         print (patterns)
        pattern_counter = Counter()
        for word in vocab:
            for second_word in vocab:
                if word != second_word:
                    extract_patterns_in_words(patterns,pattern_counter,word,second_word,max_len)
        pickle.dump(patterns, patterns_file_w)
        patterns_file_w.close()
        pickle.dump(pattern_counter, pattern_counter_file_w)
        pattern_counter_file_w.close()
    return patterns, pattern_counter


def downsample_patterns():
    #Downsample to include only top 1000
    pattern_1000 = defaultdict(list)
    for pattern,items in patterns.items():
        shuffle(items)
        pattern_1000[pattern] = items[:1000]
    return pattern_1000


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


def pair_wise_similarity(word_pair1, word_pair2,annoy_index=None, topn = 10,):
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


def get_hit_rate(patterns, similarity_function):
    if os.path.exists('data/hitrate_'+ str(len(word_vectors.vocab))):
        hit_rate_file_r = open('data/hitrate_'+ str(len(word_vectors.vocab)), 'rb')
        hit_rates_rules = pickle.load(hit_rate_file_r)
        return hit_rates_rules
    else:
        hit_rate_file_w = open('data/hitrate_'+ str(len(word_vectors.vocab)),"wb" )
        hit_rates_rules = {}
        for (pattern,support_set) in patterns.items():
            hit_rates_word_pair = {}
            for pair1 in support_set:
                hit_count = 0
                hit_pairs = set()
                for pair2 in support_set:
                    if pair1 != pair2 and similarity_function(pair1, pair2, 10):
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


def update_morpho_rules(patterns, morphological_rules = {}):
    cons = 4
    for pattern in patterns:
        transformations = patterns[pattern]
        support_set = set(sampled_patterns[pattern])
        while True:
            transformations_by_count = sorted(transformations.items(), key=lambda kv: len(kv[1]), reverse=True)
            best = transformations_by_count[0]
            if len(best[1]) >= cons:
                morphological_rules[best[0]] = (pattern, len(best[1]) / float(len(support_set)),  best[1])
                del transformations[best[0]]
            else:
                break

            #Remove all explained pairs from the support set
            #TODO: Remove best[0] from support set and transformations
            support_set = support_set - best[1]
            for k, v in transformations.items():
                transformations[k] = transformations[k] - best[1]

            transformations_by_count.pop(0)
            if not (len(support_set) >= cons and len(transformations_by_count) and len(transformations_by_count[0][1]) >= cons):
                break
    return morphological_rules


def build_graph(G):
    for dw,support in morphological_rules.items():
        morp_rule, hit_rate,support_set = support
        (word1, word2) = dw
        for (word3, word4) in support_set:
            (rank,cos_sim) = get_similarity_rank((word1,word2),(word3,word4))
            if rank < 3 and cos_sim > 0.5:
                G.add_edge(word3,word4,dw=dw,cos=0.35,rank=1)
                if not G.has_edge(word3,word4,key=dw):
                    G.add_edge(word3,word4,key=dw,cos=cos_sim,rank=rank)
            else:
                pass
    return G


def normalize_graph(G):
    for node in list(G.nodes):
        for neighbor in list(G.neighbors(node)):
            if word_vectors.vocab[node].count > word_vectors.vocab[neighbor].count:
                if G.has_edge(node, neighbor):
                    G.remove_edges_from(set(G.in_edges(neighbor, keys=True)) and set(G.out_edges(node, keys=True)))
                if G.number_of_edges(neighbor, node) > 1:
                    #                 print (list(G.in_edges(node,keys=True)))
                    n_list = [(G[neighbor][node][item]['rank'], G[neighbor][node][item]['cos'], item) for item in
                              (G[neighbor][node].keys())]
                    min_rank_edge = min(n_list, key=itemgetter(0))
                    max_cos_edge = max(n_list, key=itemgetter(1))
                    #                 print (list(G.in_edges(node,keys=True)))
                    remove_edges = [x for x in list(G.in_edges(node, keys=True)) if
                                    x != (neighbor, node, min_rank_edge[2])]
                    #                 print (remove_edges)
                    G.remove_edges_from(remove_edges)
                    if G.number_of_edges(neighbor, node) > 1:
                        remove_edges = [x for x in list(G.in_edges(node, keys=True)) if
                                        x != (neighbor, node, max_cos_edge[2])]
                        G.remove_edges_from(remove_edges)
                        print(list(G.in_edges(node, keys=True)))
            else:
                if G.has_edge(neighbor, node):
                    G.remove_edges_from(set(G.in_edges(node, keys=True)) and set(G.out_edges(neighbor, keys=True)))
                if G.number_of_edges(node, neighbor) > 1:
                    n_list = [(G[node][neighbor][item]['rank'], G[node][neighbor][item]['cos'], item) for item in
                              (G[node][neighbor].keys())]
                    min_rank_edge = min(n_list, key=itemgetter(0))
                    max_cos_edge = max(n_list, key=itemgetter(1))
                    remove_edges = [x for x in list(G.in_edges(neighbor, keys=True)) if
                                    x != (node, neighbor, min_rank_edge[2])]
                    G.remove_edges_from(remove_edges)
                    if G.number_of_edges(node, neighbor) > 1:
                        remove_edges = [x for x in list(G.in_edges(neighbor, keys=True)) if
                                        x != (node, neighbor, max_cos_edge[2])]
                        G.remove_edges_from(remove_edges)

    return G


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    logging.info("\n\n\nLoading Embeddings..")
    word_vectors = KeyedVectors.load_word2vec_format('/home/raja/models/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=1000)
    logging.info("Length of the Vocab: %s", len(word_vectors.vocab))

    logging.info("Building patterns..")
    patterns, pattern_counter = build_pattern_dict(word_vectors.vocab.keys())

    logging.info("Downsampling patterns..")
    sampled_patterns = downsample_patterns()

    logging.info ("Getting hit rates")
    hit_rates = get_hit_rate(sampled_patterns, pair_wise_similarity)

    logging.info ("Extracting morpho rules")
    morphological_rules = update_morpho_rules(hit_rates)

    logging.info("Building graph from morpho rules")
    G = nx.MultiDiGraph()
    G.add_nodes_from(word_vectors.vocab.keys())

    logging.info("Finished adding nodes")
    G = build_graph(G)

    logging.info("Finsihed building graph")
    G = normalize_graph(G)

    logging.info("graph normalized")

