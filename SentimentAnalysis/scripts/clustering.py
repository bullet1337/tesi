from collections import defaultdict
from operator import attrgetter
from typing import Set

from scripts.typez_and_constants import Node, Polarity, WeightedEdge, sgn, time_wrap


def distance(node: Node, graph: defaultdict(Set[WeightedEdge])):
    assert node in graph

    pos_dist = neg_dist = 0
    pos_connected = neg_connected = False
    for edge in graph[node]:
        if edge.node.polarity.value > 0:
            pos_dist += edge.weight
            pos_connected = True
        elif edge.node.polarity.value < 0:
            neg_dist += edge.weight
            neg_connected = True
    if pos_connected and neg_connected:
        return pos_dist - neg_dist, True
    else:
        return (pos_dist or 0) - (neg_dist or 0), False


@time_wrap()
def clusterize(graph):
    neutral_connected = set()
    for node in graph:
        if node.polarity != Polarity.NEUTRAL:
            for edge in graph[node]:
                if edge.node.polarity == Polarity.NEUTRAL:
                    neutral_connected.add(edge.node)

    both_connected = []
    single_connected = []
    marked = 0
    visited = set()
    while neutral_connected:
        both_connected.clear()
        single_connected.clear()
        for node in neutral_connected:
            node.weight, node.both_connected = distance(node, graph)
            if node.both_connected:
                both_connected.append(node)
            else:
                single_connected.append(node)

        neutral_select = sorted(both_connected or single_connected, key=lambda x: abs(x.weight), reverse=True)
        if len(neutral_select) > 1 and neutral_select[0] == neutral_select[1]:
            temp = []
            for v in neutral_select:
                if abs(v.weight) == neutral_select[0].weight:
                    temp.append(v)
                else:
                    break
            neutral_select = temp

            for v in neutral_select:
                v.impact = 0
                for edge in graph[v]:
                    if edge.node in neutral_connected:
                        v.impact += sgn(v.weight) * sgn(edge.node.weight) * edge.weight

            neutral_select = sorted(neutral_select, key=attrgetter('impact'), reverse=True)

        node = neutral_select[0]
        node.polarity = Polarity(sgn(node.weight) * 2)
        marked += 1
        print(node)
        visited.add(node)
        for edge in graph[node]:
            if edge.node.polarity == Polarity.NEUTRAL and edge.node not in visited:
                neutral_connected.add(edge.node)
        neutral_connected.remove(node)
    return marked
