#!/usr/bin/python3
import importlib
import shutil
from collections import Counter

from scripts.clustering import clusterize
from scripts.dependency_utils import get_dep_tree_parallel
from scripts.adj_graph_building import generate_adj_graph_par
from scripts.adj_graph_building import generate_adj_graph_par
from scripts.morpho_utils import lemmatize_and_tag_parallel
from scripts.ner import get_entities, merge_entities_par
from scripts.postprocessing import set_tokens_polarity, link_ta_to_ne_par, adapt_polarity_par
from scripts.text_utils import normalize_parallel, segmentize, tokenize_old
from scripts.typez_and_constants import *
from scripts.utils import *

STATS_PATH = BASE_DIR + '/{}_stats.tsv'
DICT_PATH = BASE_DIR + '/configs/dict/%s.tsv'
BASE_GRAPHS_DIR = BASE_DIR + '/graphs/%s'
BASE_TEXT_PATH = BASE_DIR + '/texts/%s.txt'
BASE_RES_PATH = BASE_DIR + '/html/%s.html'
DICTS_PATH = BASE_DIR + '/dicts/{}_{}.tsv'

EXTRACT = False
PREFIX_MODE = PrefixHandle.WEAK
STRICT_EE = False
if __name__ == '__main__':
    stats = {}
    ONLY_STATS = False
    ONLY_DICT = True
    open(STATS_PATH.format('text'), mode='w', encoding='utf8').close()
    open(STATS_PATH.format('pos'), mode='w', encoding='utf8').close()
    open(STATS_PATH.format('graph'), mode='w', encoding='utf8').close()

    for category in CATEGORIES:
        if category in ['test',]:
            continue

        print()
        stats[category] = {}
        TEXT_PATH = BASE_TEXT_PATH % category
        GRAPHS_DIR = BASE_GRAPHS_DIR % category
        RES_PATH = BASE_RES_PATH % category

        stats[category]['count'] = len(open(TEXT_PATH, mode='r', encoding='utf8').readlines())
        print('{}: {}'.format(CATEGORIES[category], stats[category]['count']))

        exec_time_millis = round(time.time() * 1000)

        # Splitting into sentences
        sentences = segmentize(TEXT_PATH)
        # Tokenizing
        tokenize_old(sentences, TEXT_PATH)
        sentences = [sentence for sentence in sentences if len(sentence.word_ids) > 0]
        sentences_count = len(sentences)
        stats[category]['s_count'] = len(sentences)
        stats[category]['avg_s'] = round(stats[category]['s_count'] / stats[category]['count'])
        # Normalizing tokens
        normalize_parallel(sentences)
        stats[category]['w_count'] = sum([len(sentence.word_ids) for sentence in sentences])
        stats[category]['avg_w'] = round(stats[category]['w_count'] / stats[category]['count'])
        with open(STATS_PATH.format('text'), mode='a', encoding='utf8') as stats_file:
            stats_file.write('\t'.join(['{category}', '{count}', '{s_count} / {avg_s}', '{w_count} / {avg_w}']).format(
                **stats[category], category=CATEGORIES[category]
            ))
            stats_file.write('\n')

        # Tagging
        lemmatize_and_tag_parallel(sentences)

        stats[category].update(get_pos_stats(sentences, stats[category]['w_count']))
        with open(STATS_PATH.format('pos'), mode='a', encoding='utf8') as stats_file:
            stats_file.write('\t'.join(['{category}', '{V}', '{S}', '{A}', '{ADV}', '{CONJ}']).format(
                **stats[category], category=CATEGORIES[category]
            ))
            stats_file.write('\n')

        if ONLY_STATS:
            continue

        # MaltParser
        get_dep_tree_parallel(sentences)

        # Extracting NE
        get_entities(sentences)
        # Merging same NE
        merge_entities_par(sentences)

        if os.path.exists(GRAPHS_DIR):
            shutil.rmtree(GRAPHS_DIR)
        os.makedirs(GRAPHS_DIR)

        [print_depenency_tree(sentence, GRAPHS_DIR) for sentence in sentences]

        sent_dict = {}
        exceptions = {}
        graph = {}
        stats[category]['new'] = 0
        for pos in ['adv', 'adj', 'noun', 'verb']:
            # sentiment dict
            sent_dict[pos], exceptions[pos] = get_dict(DICT_PATH % pos, STRICT_EE)
            if pos == 'verb':
                for node in graph['adj']:
                    if node.polarity != Polarity.NEUTRAL:
                        sent_dict[pos][node.text] = node.polarity

            # Sentiment words graph building
            graph[pos] = getattr(importlib.import_module('scripts.' + pos + '_graph_building'),
                                 'generate_' + pos + '_graph_par')\
                (sentences, sent_dict[pos], exceptions[pos], PREFIX_MODE, STRICT_EE)
            # stats_string += '%d\t%d\t%d\t' % (len(graph[pos]),
            #                                   len([node for node in graph[pos] if not graph[pos][node]]),
            #                                   len([node for node in graph[pos] if node.polarity.value]))
            print_sentiment_graph(graph[pos], BASE_DIR + '/res', '%s_%s_unmarked' % (pos, category), False)

            # Clustering neutral words in POS and NEG sets
            stats[category]['new'] += clusterize(graph[pos])
            # graph = gc1.build_nx_graph(graph)
            for node in graph[pos]:
                print(node)

            # stats_string += '%d\n' % 0 # marked
            print_sentiment_graph(graph[pos], BASE_DIR + '/res', '%s_%s_marked' % (pos, category), False)

            # Postprocessing
            set_tokens_polarity(graph[pos])

            with open(DICTS_PATH.format(category, pos), mode='w', encoding='utf8') as file:
                for w in graph[pos]:
                    if w not in sent_dict[pos]:
                        file.write('{}\t{}\n'.format(w.text, w.polarity.value))
        if ONLY_DICT:
            continue

        stats[category]['nodes'] = sum(len(g) for g in graph.values())
        stats[category]['single'] = sum(len([v for v in g.values() if len(v) == 0]) for g in graph.values())
        stats[category]['dict'] = sum(len([n for n in g if n.text in sent_dict[pos]]) for pos, g in graph.items())

        with open(STATS_PATH.format('graph'), mode='a', encoding='utf8') as stats_file:
            stats_file.write('\t'.join(['{category}', '{nodes}', '{single}', '{dict}', '{new}']).format(
                **stats[category], category=CATEGORIES[category]
            ))
            stats_file.write('\n')

        # Adapting polarities
        adapt_polarity_par(sentences)
        # Linking TA with NE
        link_ta_to_ne_par(sentences)

        [print_depenency_tree(sentence, GRAPHS_DIR) for sentence in sentences]

        for sentence in sentences:
            sentence.interesting = len(sentence.facts) > 0
        print('{}: {} FACTS'.format(category, sum([len(sentence.facts) for sentence in sentences])))

        with open(RES_PATH, mode='w', encoding='utf8') as file:
            file.write(render(BASE_DIR + '/html/template.html', {'sentences': sentences, 'category': category}))

        exec_time_millis = round(time.time() * 1000) - exec_time_millis

        print('[ TOTAL ] COMPLETED IN %d MIN %d SEC %d MSEC' % (exec_time_millis / 1000 / 60,
                                                                exec_time_millis / 1000,
                                                                exec_time_millis % 1000))
