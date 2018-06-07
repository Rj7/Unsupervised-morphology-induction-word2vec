import argparse
import itertools
import logging
import os
import pickle
import shelve
import sys
from collections import Counter
from multiprocessing import Manager, Pool
from operator import itemgetter
from random import shuffle

import networkx as nx
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename='morph_induction.log')

parser = argparse.ArgumentParser(description='Unsupervised morphology induction')
parser.add_argument('-e', '--embedding', type=str, choices=['glove', 'w2v', 'fasttext'],
                    default='w2v',
                    help="The type of word embeddings to use for morphology induction.")
parser.add_argument('-v', '--vocab_size', type=int, default=None,
                    help="Vocabulary size to extract morphology from.")
parser.add_argument('-d', '--data_dir', type=str, default='data',
                    help="Data directory to store and load files")

opts = parser.parse_args()
data_dir = opts.data_dir
embedding_file = ''
is_binary_embedding = True
if opts.embedding == 'glove':
    embedding_file = '/home/raja/models/glove50.txt'
    is_binary_embedding = False
elif opts.embedding == 'w2v':
    embedding_file = '/home/raja/models/GoogleNews-vectors-negative300.bin'
    is_binary_embedding = True
elif opts.embedding == 'fasttext':
    embedding_file = '/home/raja/models/wiki-news-300d-1M.vec'
    is_binary_embedding = False

logging.info("\n\n\nLoading Embeddings: %s", embedding_file)
word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=is_binary_embedding)

if opts.vocab_size:
    VOCAB_SIZE = opts.vocab_size
else:
    VOCAB_SIZE = len(word_vectors.vocab)

vocab_counter = Counter()
for eng_word in word_vectors.vocab.keys():
    vocab_counter[eng_word] = word_vectors.vocab[eng_word].count

# Get only top n words based on count to build morphology transformation.
vocab_words = [k for (k, v) in vocab_counter.most_common(VOCAB_SIZE)]

MIN_EXPLAINS_COUNT = 4
MIN_RANK = 3
MIN_COS = 0.5
MAX_LEN = 6


def extract_patterns_in_words(patterns, word1, word2, max_len):
    i = 1
    while word1[:i] == word2[:i]:
        i = i + 1
    if i != 1 and i > max(len(word1[i - 1:]), len(word2[i - 1:])) < max_len:
        if ("suffix", word1[i - 1:], word2[i - 1:]) in patterns:
            patterns[("suffix", word1[i - 1:], word2[i - 1:])].append((word1, word2))
        else:
            patterns[("suffix", word1[i - 1:], word2[i - 1:])] = [(word1, word2)]
            #         patterns[("suffix",word1[i-1:], word2[i-1:], word1, word2)] += 1
    i = 1
    while word1[-i:] == word2[-i:]:
        i = i + 1
    if i != 1 and max(len(word1[:-i + 1]), len(word2[:-i + 1])) < max_len:
        if ("prefix", word1[:-i + 1], word2[:-i + 1]) in patterns:
            patterns[("prefix", word1[:-i + 1], word2[:-i + 1])].append((word1, word2))
        else:
            patterns[("prefix", word1[:-i + 1], word2[:-i + 1])] = [(word1, word2)]
            #         patterns[("prefix",word1[:-i+1], word2[:-i+1], word1, word2)] += 1
    return patterns


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def parallel_build_pattern(vocab_chunk, i, vocab=vocab_words):
    patterns = {}
    for first_word in vocab_chunk:
        for second_word in vocab:
            if first_word != second_word:
                extract_patterns_in_words(patterns, first_word, second_word, MAX_LEN)
    pattern_chunk_file_w = data_dir + '/patterns/pattern_chunk_' + str(i)
    with open(pattern_chunk_file_w, 'wb') as f:
        logging.info("Writing Results to file %s", pattern_chunk_file_w)
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(patterns, f, pickle.HIGHEST_PROTOCOL)
    del patterns


def build_pattern_dict():
    patterns_dict_file = data_dir + '/sampled_patterns_' + str(VOCAB_SIZE)
    if os.path.exists(patterns_dict_file + '.dat'):
        logging.info("Loading patterns from file: %s", patterns_dict_file)
        patterns = shelve.open(patterns_dict_file)
        return patterns
    else:
        logging.info("Creating patterns")
        # patterns  = {}
        # for word in vocab:
        #     for second_word in vocab:
        #         if word != second_word:
        #             extract_patterns_in_words(patterns,word,second_word,max_len)
        with Pool() as pool:
            # Split vocab into 100 chunks to run pattern building in parallel
            vocab_chunks = (chunks(vocab_words, int(VOCAB_SIZE / 100)))
            jobs = ((vocab_chunk, i) for i, vocab_chunk in enumerate(vocab_chunks))
            pool.starmap(parallel_build_pattern, jobs, chunksize=pool._processes)
            # job_results_file_w = open(data_dir + '/job_result_patterns_' + str(len(vocab)), "wb")
            # logging.info("Writing Results to file")
            # pickle.dump(job_results, job_results_file_w )
            # job_results_file_w.close()

        logging.info("Merging Results")
        patterns_dir = data_dir + '/patterns/'
        patterns = shelve.open(patterns_dict_file)
        for filename in os.listdir(patterns_dir):
            with open(patterns_dir + filename, 'rb') as handle:
                result = pickle.load(handle)
                for key in result.keys():
                    if repr(key) in patterns:
                        patterns[repr(key)] = result[key] + patterns[repr(key)]
                    else:
                        patterns[repr(key)] = result[key]
        logging.info("length of pattern: %s", len(patterns))
        logging.info("Saved patterns dict")
        logging.info("Downsampling patterns..")
        return downsample_patterns(patterns)


def downsample_patterns(patterns):
    # Downsample to include only top 1000
    for pattern, items in patterns.items():
        shuffle(items)
        patterns[pattern] = items[:1000]
    logging.info("Downsampled patterns dict")
    return patterns


def pair_wise_similarity(word_pair1, word_pair2, topn=10):
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn)
    #     print (word_pair2[1])
    #     print (closest_n)
    for word, cos_sim in closest_n:
        if word == word_pair2[1]:
            return True
    return False


def annoy_pair_wise_similarity(word_pair1, word_pair2, indexer, topn=10):
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn,
                                          indexer=indexer)
    #     print (word_pair2[1])
    #     print (closest_n)
    for word, cos_sim in closest_n:
        if word == word_pair2[1]:
            return True
    return False


def get_similarity_rank(word_pair1, word_pair2, similarity_dict):
    topn = 500
    closest_n = word_vectors.most_similar(positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn,
                                          indexer=annoy_index)
    #     print (word_pair2[1])
    #     print (closest_n)
    outside_topn = True
    for n, (word, cos_sim) in enumerate(closest_n):
        if word == word_pair2[1]:
            outside_topn = False
            similarity_dict[word_pair1, word_pair2] = (n, cos_sim)
    if outside_topn:
        similarity_dict[word_pair1, word_pair2] = (topn, 0)


def get_hit_rate(patterns, similarity_function, indexer=None):
    if os.path.exists('data/hitrate_' + str(len(word_vectors.vocab))):
        hit_rate_file_r = open(data_dir + '/hitrate_' + str(VOCAB_SIZE), 'rb')
        hit_rates_rules = pickle.load(hit_rate_file_r)
        hit_rate_file_r.close()
        return hit_rates_rules
    else:
        hit_rates_rules = {}
        for (pattern, support_set) in patterns.items():
            hit_rates_word_pair = {}
            for pair1 in support_set:
                hit_count = 0
                hit_pairs = set()
                for pair2 in support_set:
                    if pair1 != pair2 and similarity_function(pair1, pair2, indexer, 10):
                        hit_count += 1
                        hit_pairs.add(pair2)
                if hit_count != 0:
                    hit_rates_word_pair[pair1] = hit_pairs
            if len(support_set) != 1 and hit_rates_word_pair:
                hit_rates_rules[pattern] = hit_rates_word_pair
        hit_rate_file_w = open(data_dir + '/hitrate_' + str(VOCAB_SIZE), "wb")
        pickle.dump(hit_rates_rules, hit_rate_file_w)
        hit_rate_file_w.close()
        return hit_rates_rules


def get_annoy(w2v, embedding_type='w2v'):
    dims = 100
    annoy_file_name = data_dir + '/annoy_index_' + '_' + str(dims) + '_' + embedding_type + '_' + str(len(w2v.vocab))
    if os.path.exists(annoy_file_name):
        logging.info("Loading Annoy from file: %s", annoy_file_name)
        nn_index = AnnoyIndexer()
        nn_index.load(annoy_file_name)
        nn_index.model = word_vectors
    else:
        logging.info("Creating Annoy")
        nn_index = AnnoyIndexer(word_vectors, dims)
        nn_index.save(annoy_file_name)
        logging.info("Annoy indexing saved to %s", annoy_file_name)
    return nn_index


def get_hit_rules(pattern, support_set, hit_rates_rules):
    hit_rates_word_pair = {}
    for pair1 in support_set:
        hit_count = 0
        hit_pairs = set()
        for pair2 in support_set:
            if pair1 != pair2 and annoy_pair_wise_similarity(pair1, pair2, annoy_index, 10):
                hit_count += 1
                hit_pairs.add(pair2)
        if hit_count:
            hit_rates_word_pair[pair1] = hit_pairs
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
    hit_rate_file_name = data_dir + '/hitrate_' + str(vocab_size)
    if os.path.exists(hit_rate_file_name):
        logging.info("Loading hit rates from file: %s", hit_rate_file_name)
        hit_rate_r = open(hit_rate_file_name, 'rb')
        hit_rates_ = pickle.load(hit_rate_r)
        hit_rate_r.close()
        return hit_rates_
    else:
        logging.info("Creating hit rates")
        hit_rates_ = Manager().dict()
        pool = Pool()
        pool.starmap(get_hit_rules,
                     ((pattern, support_set, hit_rates_) for pattern, support_set in sampled_patterns.items()),
                     chunksize=pool._processes)
        logging.info("No of hits %s", len(hit_rates_))
        #     hit_rates = get_hit_rate(sampled_patterns, annoy_pair_wise_similarity, annoy_index)
        output_hit_rates = {}
        output_hit_rates.update(hit_rates)
        hit_rate_file_w = open(hit_rate_file_name, "wb")
        pickle.dump(output_hit_rates, hit_rate_file_w)
        hit_rate_file_w.close()
        del hit_rates
        return output_hit_rates


def update_morpho_rules(hit_rates_, sampled_patterns):
    """ Compute best direction vector(s) that explain many rules greedily.
    The recursion stops when it finds all direction vectors explains less than a predefined number of words (10)
    """
    morph_rules = {}
    for pattern in hit_rates_:
        transformations = hit_rates_[pattern]
        support_set = set(sampled_patterns[pattern])
        while True:
            transformations_by_count = sorted(transformations.items(), key=lambda kv: len(kv[1]), reverse=True)
            best = transformations_by_count[0]
            #         print (transformations_by_count)
            #         print (transformations)
            if len(best[1]) >= MIN_EXPLAINS_COUNT:
                morph_rules[best[0]] = (pattern, len(best[1]) / float(len(support_set)), best[1])
                #             directions.append(best)
                del transformations[best[0]]
            else:
                break

            # TODO: Remove all explained pairs from the support set
            # TODO: Remove best[0] from support set and transformations
            support_set = support_set - best[1]
            for k, v in transformations.items():
                #             print ("*"*50)
                #             print (transformations[k])
                transformations[k] = transformations[k] - best[1]
                #             print (transformations[k])
                #             print ("__"*50)

            transformations_by_count.pop(0)
            if not (len(support_set) >= MIN_EXPLAINS_COUNT and len(transformations_by_count) and len(
                    transformations_by_count[0][1]) >= MIN_EXPLAINS_COUNT):
                break
    logging.info("No of morphological rules: %s", len(morph_rules))
    return morph_rules


def add_graph_edges(di_graph, morph_rules):
    logging.info("Adding edges to the graph")
    similarity_dict = Manager().dict()
    jobs = (((word1, word2), (word3, word4), similarity_dict)
            for (word1, word2), (morp_rule, hit_rate, support_set) in morph_rules.items()
            for (word3, word4) in support_set)
    pool = Pool()
    pool.starmap(get_similarity_rank, jobs, chunksize=pool._processes)

    for dw, support in morph_rules.items():
        morp_rule, hit_rate, support_set = support
        (word1, word2) = dw
        for (word3, word4) in support_set:
            (rank, cos_sim) = similarity_dict[(word1, word2), (word3, word4)]
            if rank < MIN_RANK and cos_sim > MIN_COS:
                if not di_graph.has_edge(word3, word4, key=dw):
                    di_graph.add_edge(word3, word4, key=dw, cos=cos_sim, rank=rank, morp_rule=morp_rule)
            else:
                #             print (rank,cos_sim, word3, word2, dw)
                pass

    logging.info("No of nodes in graph: %s", len(di_graph.nodes))
    logging.info("No of edges in graph: %s", len(di_graph.edges))
    return di_graph


def normalize_and_save_graph(di_graph):
    for node in list(di_graph.nodes):
        for neighbor in list(di_graph.neighbors(node)):
            if word_vectors.vocab[node].count > word_vectors.vocab[neighbor].count:
                if di_graph.has_edge(node, neighbor):
                    di_graph.remove_edges_from(set(di_graph.in_edges(neighbor, keys=True))
                                               and set(di_graph.out_edges(node, keys=True)))
                if di_graph.number_of_edges(neighbor, node) > 1:
                    #                 print (list(G.in_edges(node,keys=True)))
                    n_list = [(di_graph[neighbor][node][item]['rank'], di_graph[neighbor][node][item]['cos'], item)
                              for item in (di_graph[neighbor][node].keys())]
                    min_rank_edge = min(n_list, key=itemgetter(0))
                    max_cos_edge = max(n_list, key=itemgetter(1))
                    #                 print (list(G.in_edges(node,keys=True)))
                    remove_edges = [x for x in list(di_graph.in_edges(node, keys=True)) if
                                    x != (neighbor, node, min_rank_edge[2])]
                    #                 print (remove_edges)
                    di_graph.remove_edges_from(remove_edges)
                    if di_graph.number_of_edges(neighbor, node) > 1:
                        remove_edges = [x for x in list(di_graph.in_edges(node, keys=True)) if
                                        x != (neighbor, node, max_cos_edge[2])]
                        di_graph.remove_edges_from(remove_edges)
            else:
                if di_graph.has_edge(neighbor, node):
                    di_graph.remove_edges_from(set(di_graph.in_edges(node, keys=True)) and
                                               set(di_graph.out_edges(neighbor, keys=True)))
                if di_graph.number_of_edges(node, neighbor) > 1:
                    n_list = [(di_graph[node][neighbor][item]['rank'], di_graph[node][neighbor][item]['cos'], item)
                              for item in (di_graph[node][neighbor].keys())]
                    min_rank_edge = min(n_list, key=itemgetter(0))
                    max_cos_edge = max(n_list, key=itemgetter(1))
                    remove_edges = [x for x in list(di_graph.in_edges(neighbor, keys=True)) if
                                    x != (node, neighbor, min_rank_edge[2])]
                    di_graph.remove_edges_from(remove_edges)
                    if di_graph.number_of_edges(node, neighbor) > 1:
                        remove_edges = [x for x in list(di_graph.in_edges(neighbor, keys=True)) if
                                        x != (node, neighbor, max_cos_edge[2])]
                        di_graph.remove_edges_from(remove_edges)

    logging.info("No of nodes in graph: %s", len(di_graph.nodes))
    logging.info("No of edges in graph: %s", len(di_graph.edges))

    norm_graph_file = data_dir + '/norm_graph_' + str(len(di_graph.nodes)) + '_' + str(len(di_graph.edges))
    normalized_graph_w = open(norm_graph_file, "wb")
    pickle.dump(di_graph, normalized_graph_w)
    normalized_graph_w.close()

    logging.info("Saved graph file to %s", norm_graph_file)
    return di_graph


if __name__ == '__main__':
    logging.info("Settings: %s", opts)
    logging.info("Getting patterns..")
    downsampled_patterns = build_pattern_dict()

    logging.info("Getting annoyed")
    annoy_index = get_annoy(word_vectors, opts.embedding)

    logging.info("Getting hit rates")
    hit_rates = get_hit_rates(downsampled_patterns, VOCAB_SIZE)

    logging.info("Getting Morphological rules")
    morphological_rules = update_morpho_rules(hit_rates, downsampled_patterns)

    logging.info("Building Graph")
    di_multi_graph = nx.MultiDiGraph()

    logging.info("Added nodes and edged to Graph")
    di_multi_graph.add_nodes_from(vocab_words)
    di_multi_graph = add_graph_edges(di_multi_graph, morphological_rules)

    logging.info("Normalizing graph based on count")
    normalize_and_save_graph(di_multi_graph)
    logging.info("END!!")

    sys.exit()
