import copy
from collections import defaultdict
from typing import List, Set

from joblib import Parallel
from joblib import delayed

from scripts.paths_extracting import get_paths_by_conjuctive
from scripts.paths_extracting import get_paths_by_determinative
from scripts.polarity import get_polarity
from scripts.typez_and_constants import Node, Polarity, Token, Sentence, PrefixHandle, WeightedEdge, sgn, NUM_CORES, \
    dd, Orientation


def add_edge(from_node: Node, to_node: Node, graph: defaultdict(set), inverse: bool=False):
    if from_node == to_node:
        return

    polarity = inverse ^ ((from_node.polarity.value < 0) == (to_node.polarity.value < 0))
    edge = WeightedEdge(to_node)
    for graph_edge in graph[from_node]:
        if edge == graph_edge:
            graph_edge.polarities[polarity].append((from_node.polarity, to_node.polarity))

            edge = WeightedEdge(from_node)
            for graph_other_edge in graph[to_node]:
                if edge == graph_other_edge:
                    graph_other_edge.polarities[polarity].append((to_node.polarity, from_node.polarity))
            return

    graph_from = None
    graph_to = None
    for key in graph:
        if key == from_node:
            graph_from = key
        if key == to_node:
            graph_to = key

    graph[from_node].add(WeightedEdge(graph_to, polarity, (from_node.polarity, to_node.polarity)))
    graph[to_node].add(WeightedEdge(graph_from, polarity, (to_node.polarity, from_node.polarity)))
    return


def process(path: List[List[Node]], graph: defaultdict(set)):
    if len(path) == 1 and len(path[0]) == 1:
        return

    for lidx, level in enumerate(path):
        temp_level = level.copy()
        while temp_level:
            node = temp_level.pop()
            if not node.process:
                continue

            # and local
            for other_node in temp_level:
                if not other_node.process:
                    continue

                add_edge(node, other_node, graph)
            # but global
            inverse = True
            for next_level in path[lidx + 1:]:
                for next_node in next_level:
                    if not next_node.process:
                        continue

                    add_edge(node, next_node, graph, inverse)
                inverse = not inverse


def weigh(graph: defaultdict(set)):
    for _, edges in graph.items():
        for edge in edges:
            edge.weight = len(edge.polarities[True]) - len(edge.polarities[False])


def init_graph(sentence: Sentence, graph, node_filter,
               entries, mode: PrefixHandle, strict_ee: bool,
               words: Set[str], exceptions: Set[str], deps_handler):
    token_node_map = {}
    for token in sentence.tokens:
        if node_filter(token):
            node = Node(token, strict_ee)
            get_polarity(node, token, sentence, mode, words, exceptions, deps_handler)
            entries[node][sentence].append(token.id)
            graph[node]
            token_node_map[token] = node
    return token_node_map


def tokens_2_nodes(token_paths: List[List[List[Token]]], token_nodes_map):
    nodes = []
    for tokend_path in token_paths:
        path = []
        for group in tokend_path:
            temp = []
            for token in group:
                temp.append(token_nodes_map[token])
            path.append(temp)
        nodes.append(path)
    return nodes


def init_graph_polarity(graph):
    for node in graph:
        if node.polarity == Polarity.NEUTRAL:
            init_polarity = None
            for edge in graph[node]:
                if edge.node.polarity == Polarity.NEUTRAL:
                    break
                elif init_polarity is None:
                    init_polarity = edge.node.polarity
                    node.weight += edge.weight
                else:
                    if init_polarity == edge.node.polarity:
                        node.weight += edge.weight
                    else:
                        break
            else:
                if init_polarity is not None:
                    node.polarity = Polarity(sgn(node.weight) * init_polarity.value)


def sort(splitted_paths: List[List[List[Node]]]):
    topology = defaultdict(dd)
    for pidx, path in enumerate(splitted_paths[:]):
        topology[pidx]
        start = path[0]
        for other_pidx, other_path in enumerate(splitted_paths[pidx + 1:]):
            # intersection between path's first node and other path
            # there is only only one IN path
            found = False
            for elem in start:
                for ogidx, other_group in enumerate(other_path):
                    for other_elem in other_group:
                        if elem == other_elem \
                                and sgn(elem.polarity.value) == sgn(other_elem.polarity.value)\
                                and elem.process == other_elem.process:
                            topology[pidx][Orientation.IN].append((other_pidx + pidx + 1, 0))
                            topology[other_pidx + pidx + 1][Orientation.OUT].append((pidx, ogidx))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            # intersection between other path's first node and path
            # many OUT paths allowed
            if not found:
                other_start = other_path[0]
                for elem in other_start:
                    found = False
                    for gidx, group in enumerate(path):
                        for group_elem in group:
                            if elem == group_elem \
                                    and sgn(elem.polarity.value) == sgn(group_elem.polarity.value)\
                                    and elem.process == group_elem.process:
                                topology[other_pidx + pidx + 1][Orientation.IN].append((pidx, 0))
                                topology[pidx][Orientation.OUT].append((other_pidx + pidx + 1, gidx))
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

    if not topology:
        return None, None

    for k in topology:
        if Orientation.IN not in topology[k] or len(topology[k][Orientation.IN]) == 1:
            print('WARNING')

    print('TOPOLOGY')
    for k, v in topology.items():
        print(splitted_paths[k], v)

    toplogy_order = []
    toplogy_copy = copy.deepcopy(topology)
    while toplogy_copy:
        picked = None
        for path in toplogy_copy:
            if not toplogy_copy[path][Orientation.IN]:
                toplogy_order.append(path)
                picked = path
                break
        if not picked:
            picked = list(toplogy_copy.keys())[0]
        for adjacent_path, _ in topology[picked][Orientation.OUT]:
            toplogy_copy[adjacent_path][Orientation.IN].remove((picked, 0))
        toplogy_copy.pop(picked)
    return toplogy_order, topology


def merge(splitted_paths: List[List[List[Node]]]):
    if len(splitted_paths) == 1:
        print('SIMPLE PATH')
        return splitted_paths

    for path in splitted_paths:
        print('GROUPS ', [[elem for elem in group] for group in path])

    toplogy_order, topology = sort(splitted_paths)
    if not topology:
        return splitted_paths

    for pidx in toplogy_order:
        print(splitted_paths[pidx])

    roots = set()
    for pidx in toplogy_order:
        if Orientation.IN not in topology[pidx]:
            roots.add(pidx)
        else:
            path = splitted_paths[pidx]
            for adjacent_pidx, lidx in topology[pidx][Orientation.IN]:
                in_path = splitted_paths[adjacent_pidx]
                for gidx, group in enumerate(path):
                    if lidx < len(in_path):
                        in_path[lidx].extend(group)
                    else:
                        in_path.extend(path[gidx:])
                    lidx += 1

    paths_tree = [splitted_paths[root] for root in roots]
    for path in paths_tree:
        print('MERGED ', [[node for node in level] for level in path])
    print()
    return paths_tree


def set_groups(paths, sentence: Sentence):
    for path in paths:
        for group in path:
            for token in group:
                sentence.tokens[token.id].group.extend(sentence.tokens[t.id] for t in group if t != token)
                sentence.tokens[token.id].native_group = sentence.tokens[token.id].group[:]
                sentence.tokens[token.id].group.extend(sentence.tokens[t.id] for g in path for t in g if g != group)


def generate_graph(sentence: Sentence, tn_map,
                   node_filter, pos_conj_transition, conj_pos_transition, pos_pos_transition,
                   edge_filter, pos_pos_filter):
    paths = []
    if node_filter and pos_conj_transition and conj_pos_transition and pos_conj_transition:
        paths.extend(get_paths_by_conjuctive(sentence, node_filter, pos_conj_transition, conj_pos_transition,
                                             pos_pos_transition))
    if node_filter and edge_filter and pos_pos_filter:
        paths.extend(get_paths_by_determinative(sentence, node_filter, edge_filter, pos_pos_filter))

    if paths:
        return paths, merge(tokens_2_nodes(paths, tn_map))


def generate_graph_par(sentences: List[Sentence], sent_dict: defaultdict(set), exceptions: Set[str],
                       mode: PrefixHandle, strict_ee: bool,
                       node_filter, pos_conj_transition, conj_pos_transition, pos_pos_transition,
                       edge_filter, pos_pos_filter, deps_handler, pos: str):
    graph = defaultdict(set)
    entries = defaultdict(dd)
    total_paths = []

    words = {token.lemma.lower() for sentence in sentences for token in sentence.tokens if node_filter(token)}
    if not strict_ee:
        words = {w.replace('ё', 'е') for w in words}

    sentence_tn_map = {}
    for sentence in sentences:
        sentence_tn_map[sentence] = init_graph(sentence, graph, node_filter,
                                               entries, mode, strict_ee, words, exceptions,
                                               deps_handler)

    sentence_results = Parallel(n_jobs=NUM_CORES)(delayed(generate_graph)(
        sentence, sentence_tn_map[sentence],
        node_filter, pos_conj_transition, conj_pos_transition, pos_pos_transition, edge_filter, pos_pos_filter
    ) for sentence in sentences)

    for id, result in enumerate(sentence_results):
        if result:
            print('SENTENCE %d: %s' % (id, sentences[id].text))
            set_groups(result[0], sentences[id])
            for path in result[0]:
                print('PATH ', [elem for elem in path])
            total_paths.extend(result[1])

    for path in total_paths:
        process(path, graph)

    for node in graph:
        node.polarity = sent_dict.get(node.text, Polarity.NEUTRAL)

    for node, edges in graph.items():
        for edge in edges:
            assert edge.node.polarity is not None, edge.node
            found = False
            for other_edge in graph[edge.node]:
                if other_edge.node == node:
                    found = True
                    break
            assert found, str(node) + ' : ' + str(edge.node)

            assert True in edge.polarities and True in other_edge.polarities or \
                    True not in edge.polarities and True not in other_edge.polarities
            for p1, p2 in zip(edge.polarities[True], other_edge.polarities[True]):
                    assert p1[0] == p2[1], str(node) + ' : ' + str(edge.node)
                    assert p1[1] == p2[0], str(node) + ' : ' + str(edge.node)

            assert False in edge.polarities and False in other_edge.polarities or \
                   False not in edge.polarities and False not in other_edge.polarities
            for p1, p2 in zip(edge.polarities[False], other_edge.polarities[False]):
                    assert p1[0] == p2[1], str(node) + ' : ' + str(edge.node)
                    assert p1[1] == p2[0], str(node) + ' : ' + str(edge.node)

    weigh(graph)
    for node in graph:
        node.entries = entries[node]

    init_graph_polarity(graph)
    return graph