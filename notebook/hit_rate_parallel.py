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
import sys

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename='morph_induction.log')


logging.info ("\n\n\nLoading Embeddings..")
word_vectors = KeyedVectors.load_word2vec_format('/home/raja/models/glove50.txt', binary=False)
vocab_input = word_vectors
vocab_size = len(vocab_input.vocab)
logging.info ("Length of the Vocab: %s", vocab_size)
vocab = list(vocab_input.vocab.keys())


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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def parallel_build_pattern(vocab_chunk,i,vocab = vocab):
    patterns = {}
    MAX_LEN = 6
    for word in vocab_chunk:
        for second_word in vocab:
            if word != second_word:
                extract_patterns_in_words(patterns, word, second_word, MAX_LEN)
    pattern_chunk_file_w = '../data/patterns/pattern_chunk_' + str(i)
    with open(pattern_chunk_file_w, 'wb') as f:
        logging.info("Writing Results to file %s", pattern_chunk_file_w)
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(patterns, f, pickle.HIGHEST_PROTOCOL)
    del patterns


def build_pattern_dict():
    if os.path.exists('../data/sampled_patterns_'+ str(len(vocab))):
        logging.info("Loading patterns from file")
        patterns_file_r = open('../data/sampled_patterns_'+ str(len(vocab)), 'rb')
        sampled_patterns = pickle.load(patterns_file_r)
        patterns_file_r.close()
        return sampled_patterns
    else:
        logging.info("Creating patterns")
        # patterns  = {}
        # for word in vocab:
        #     for second_word in vocab:
        #         if word != second_word:
        #             extract_patterns_in_words(patterns,word,second_word,max_len)
        patterns = {}
        with Pool() as pool:
            # Split vocab into 100 chunks to run pattern building in parallel
            vocab_chunks = (chunks(vocab, int(len(vocab) / 100)))
            jobs = ((vocab_chunk, i) for i, vocab_chunk in enumerate(vocab_chunks))
            pool.starmap(parallel_build_pattern, jobs, chunksize=pool._processes)
            # job_results_file_w = open('../data/job_result_patterns_' + str(len(vocab)), "wb")
            # logging.info("Writing Results to file")
            # pickle.dump(job_results, job_results_file_w )
            # job_results_file_w.close()

        logging.info("Merging Results")
        patterns_dir = '../data/patterns/'
        for filename in os.listdir(patterns_dir):
            with open(patterns_dir + filename, 'rb') as handle:
                result = pickle.load(handle)
                for key in result.keys():
                    if key in patterns:
                        patterns[key] = result[key] + patterns[key]
                    else:
                        patterns[key] = result[key]
        logging.info("length of pattern: %s", len(patterns))
        logging.info("Downsampling patterns..")
        sampled_patterns = downsample_patterns(patterns)
        patterns_file_w = open('../data/sampled_patterns_' + str(len(vocab)), "wb")
        pickle.dump(sampled_patterns, patterns_file_w)
        logging.info("Saved downsampled patterns dict")
        patterns_file_w.close()
        return sampled_patterns


def downsample_patterns(patterns):
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


def get_similarity_rank(word_pair1, word_pair2, similarity_dict):
    topn = 500
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn, indexer=annoy_index)
#     print (word_pair2[1])
#     print (closest_n)
    outside_topn = True
    for n,(word, cos_sim) in enumerate(closest_n):
        if word == word_pair2[1]:
            outside_topn = False
            similarity_dict[word_pair1, word_pair2] = (n, cos_sim)
    if outside_topn:
        similarity_dict[word_pair1, word_pair2] = (topn, 0)


def get_hit_rate(patterns, similarity_function, annoy_index=None):
    if False:
        hit_rate_file_r = open('../data/hitrate_'+ str(len(vocab_input.vocab)), 'rb')
        hit_rates_rules = pickle.load(hit_rate_file_r)
        hit_rate_file_r.close()
        return hit_rates_rules
    else:
        hit_rate_file_w = open('../data/hitrate_'+ str(len(vocab_input.vocab)),"wb" )
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
    

def get_annoy(w2v, embedding_type = 'w2v'):
    dims = 100
    annoy_file_name = '../data/annoy_index_' + '_' + str(dims) + '_' + embedding_type + '_' + str(len(w2v.vocab))
    if os.path.exists(annoy_file_name):
        logging.info("Loading Annoy from file")
        annoy_index = AnnoyIndexer()
        annoy_index.load(annoy_file_name)
        annoy_index.model = word_vectors
    else:
        logging.info("Creating Annoy")
        annoy_index = AnnoyIndexer(word_vectors,dims)
        annoy_index.save(annoy_file_name)
    return annoy_index


def get_hit_rules(pattern, support_set,hit_rates_rules):
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
        hit_rates_rules[pattern] = hit_rates_word_pair

def iterator_slice(iterator, length):
    iterator = iter(iterator)
    while True:
        res = tuple(itertools.islice(iterator, length))
        if not res:
            break
        yield res


def get_hit_rates(sampled_patterns, vocab_size):
    hit_rate_file_name = '../data/hitrate_' + str(vocab_size)
    if os.path.exists(hit_rate_file_name):
        logging.info("Loading hit rates from file")
        hit_rate_r = open(hit_rate_file_name, 'rb')
        hit_rates = pickle.load(hit_rate_r)
        hit_rate_r.close()
        return hit_rates
    else:
        logging.info("Creating hit rates")
        hit_rates = Manager().dict()
        pool = Pool()
        pool.starmap(get_hit_rules,((pattern, support_set, hit_rates) for pattern,support_set in sampled_patterns.items()), chunksize = pool._processes)
        logging.info ("No of hits %s", len(hit_rates))
    #     hit_rates = get_hit_rate(sampled_patterns, annoy_pair_wise_similarity, annoy_index)
        output_hit_rates = {}
        output_hit_rates.update(hit_rates)
        hit_rate_file_w = open(hit_rate_file_name,"wb" )
        pickle.dump(output_hit_rates, hit_rate_file_w)
        hit_rate_file_w.close()
        del hit_rates
        return output_hit_rates


def update_morpho_rules(patterns, sampled_patterns):
    """ Compute best direction vector(s) that explain many rules greedily.
    The recursion stops when it finds all direction vectors explains less than a predefined number of words (10)
    """
    morphological_rules = {}
    MIN_EXPLAINS_COUNT = 4
    for pattern in patterns:
        transformations = patterns[pattern]
        support_set = set(sampled_patterns[pattern])
        while(True):
            transformations_by_count = sorted(transformations.items(), key=lambda kv: len(kv[1]), reverse=True)
            best = transformations_by_count[0]
    #         print (transformations_by_count)
    #         print (transformations)
            if len(best[1]) >= MIN_EXPLAINS_COUNT:
                morphological_rules[best[0]] = (pattern, len(best[1]) / float(len(support_set)),  best[1])
    #             directions.append(best)
                del transformations[best[0]]
            else:
                break

            #Remove all explained pairs from the support set
            #TODO: Remove best[0] from support set and transformations
            support_set = support_set - best[1]
            for k, v in transformations.items():
    #             print ("*"*50)
    #             print (transformations[k])
                transformations[k] = transformations[k] - best[1]
    #             print (transformations[k])
    #             print ("__"*50)

            transformations_by_count.pop(0)
            if not (len(support_set) >= MIN_EXPLAINS_COUNT and len(transformations_by_count) and len(transformations_by_count[0][1]) >= MIN_EXPLAINS_COUNT):
                break
    logging.info("No of morphological rules: %s", len(morphological_rules))
    return morphological_rules


def build_graph(G, morphological_rules):
    logging.info("Adding edges to the graph")
    MIN_RANK = 3
    MIN_COS = 0.5
    similarity_dict = Manager().dict()
    jobs = (((word1,word2),(word3,word4), similarity_dict)
            for (word1, word2),(morp_rule, hit_rate,support_set) in morphological_rules.items()
            for (word3, word4) in support_set)
    pool = Pool()
    pool.starmap(get_similarity_rank, jobs, chunksize=pool._processes)

    for dw,support in morphological_rules.items():
        morp_rule, hit_rate,support_set = support
        (word1, word2) = dw
        for (word3, word4) in support_set:
            (rank,cos_sim) = similarity_dict[(word1,word2),(word3,word4)]
            if rank < MIN_RANK and cos_sim > MIN_COS:
                if not G.has_edge(word3,word4,key=dw):
                    G.add_edge(word3,word4,key=dw,cos=cos_sim,rank=rank)
            else:
    #             print (rank,cos_sim, word3, word2, dw)
                pass
    
    logging.info("No of nodes in graph: %s", len(G.nodes))
    logging.info("No of edges in graph: %s", len(G.edges))
    return G


def normalize_graph(G):
    for node in list(G.nodes):
        for neighbor in list(G.neighbors(node)):
            if vocab_input.vocab[node].count > vocab_input.vocab[neighbor].count:
                if G.has_edge(node, neighbor):
                    G.remove_edges_from(set(G.in_edges(neighbor,keys=True)) and set(G.out_edges(node,keys=True)))
                if G.number_of_edges(neighbor, node) > 1:
    #                 print (list(G.in_edges(node,keys=True)))
                    n_list = [(G[neighbor][node][item]['rank'], G[neighbor][node][item]['cos'], item) for item in (G[neighbor][node].keys())]
                    min_rank_edge = min(n_list,key=itemgetter(0))
                    max_cos_edge = max(n_list,key=itemgetter(1))
    #                 print (list(G.in_edges(node,keys=True)))
                    remove_edges = [x for x in list(G.in_edges(node,keys=True)) if x != (neighbor,node,min_rank_edge[2])]
    #                 print (remove_edges)
                    G.remove_edges_from(remove_edges)
                    if G.number_of_edges(neighbor, node) > 1:
                        remove_edges = [x for x in list(G.in_edges(node,keys=True)) if x != (neighbor,node,max_cos_edge[2])]
                        G.remove_edges_from(remove_edges)
            else:
                if G.has_edge(neighbor, node):
                    G.remove_edges_from(set(G.in_edges(node,keys=True)) and set(G.out_edges(neighbor,keys=True)))
                if G.number_of_edges(node, neighbor) > 1:
                    n_list = [(G[node][neighbor][item]['rank'], G[node][neighbor][item]['cos'], item) for item in (G[node][neighbor].keys())]
                    min_rank_edge = min(n_list,key=itemgetter(0))
                    max_cos_edge = max(n_list,key=itemgetter(1))
                    remove_edges = [x for x in list(G.in_edges(neighbor,keys=True)) if x != (node,neighbor,min_rank_edge[2])]
                    G.remove_edges_from(remove_edges)
                    if G.number_of_edges(node, neighbor) > 1:
                        remove_edges = [x for x in list(G.in_edges(neighbor,keys=True)) if x != (node,neighbor,max_cos_edge[2])]
                        G.remove_edges_from(remove_edges)

    logging.info("No of nodes in graph: %s", len(G.nodes))
    logging.info("No of edges in graph: %s", len(G.edges))

    norm_graph_file = '../data/norm_graph_' + str(len(G.nodes)) + '_' + str(len(G.edges))
    normalized_graph_w = open(norm_graph_file,"wb" )
    pickle.dump(G, normalized_graph_w)
    normalized_graph_w.close()

    logging.info("Saved graph file to %s", norm_graph_file)
    return G


# word_vectors = KeyedVectors.load_word2vec_format('/home/raja/models/GoogleNews-vectors-negative300.bin.gz', binary=True)
if __name__ == '__main__':

    logging.info ("Getting patterns..")
    sampled_patterns = build_pattern_dict()

    logging.info ("Getting annoyed")
    annoy_index = get_annoy(word_vectors, 'glove')
    
    logging.info ("Getting hit rates")
    hit_rates = get_hit_rates(sampled_patterns, vocab_size)

    logging.info ("Getting Morphological rules")
    morphological_rules = update_morpho_rules(hit_rates,sampled_patterns)

    logging.info ("Building Graph")
    G = nx.MultiDiGraph()
    G.add_nodes_from(vocab_input.vocab.keys())

    logging.info ("Added nodes to Graph")
    G = build_graph(G, morphological_rules)

    logging.info ("Normalizing graph based on count")
    G = normalize_graph(G)
    logging.info ("END!!")

    sys.exit()