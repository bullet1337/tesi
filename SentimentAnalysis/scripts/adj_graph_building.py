from collections import defaultdict
from typing import List, Set

from scripts.graph_building import generate_graph_par
from scripts.polarity import get_i_polarity
from scripts.typez_and_constants import PosType, Token, Sentence, SynType, PrefixHandle, Edge, \
    interesting_feats, Node, Orientation
from scripts.utils import is_adj_or_part, is_conjunctive, is_adversative


def is_adj_adj_transition(edge: Edge, token: Token):
    return edge.type == SynType.сочин and equal_adj(edge.token, token)


def is_conj_adj_transition(edge: Edge, token: Token):
    return edge.type == SynType.соч_союзн and equal_adj(edge.token, token)


def is_adj_conj_transition(edge: Edge):
    return edge.token.pos == PosType.CONJ and edge.type == SynType.сочин \
           and (is_adversative(edge.token) or is_conjunctive(edge.token))


def is_adj_det_edge(edge: Edge):
    return edge.type == SynType.опред


def get_token_info(token: Token):
    info = {'token': token}
    feats = token.feats
    for group, group_feats in interesting_feats.items():
        for feat in group_feats:
            if feat in feats:
                assert group not in info
                info[group] = feat

        if group not in info:
            info[group] = None
    return info


def equal_adj(token1: Token, token2: Token):
    if 'plen' not in token1.feats or 'plen' not in token2.feats \
            or 'supr' in token1.feats or 'supr' in token2.feats \
            or 'comp' in token1.feats or 'comp' in token2.feats:
        return True

    return get_token_info(token1)['quantity'] == get_token_info(token2)['quantity']


def handle_adj_deps(node: Node, token: Token, sentence: Sentence):
    ampl_token = None
    for edge in sentence.dep_tree[token]:
        if edge.orientation == Orientation.IN:
            continue

        if edge.type in {SynType.огранич, SynType.колич_огран, SynType.опред} \
            and edge.token.pos in {PosType.ADV, PosType.APRO}:
                ampl_token = edge.token
                break

    if ampl_token is not None:
        node.polarity = get_i_polarity(ampl_token, node.polarity, True)
        if ampl_token.ampl_token is not None:
            node.polarity = get_i_polarity(ampl_token.ampl_token, node.polarity, True)


def generate_adj_graph_par(sentences: List[Sentence], sent_dict: defaultdict(set), exceptions: Set[str],
                           mode: PrefixHandle, strict_ee: bool):
    return generate_graph_par(sentences, sent_dict, exceptions, mode, strict_ee,
                              is_adj_or_part, is_adj_conj_transition, is_conj_adj_transition, is_adj_adj_transition,
                              is_adj_det_edge, equal_adj, handle_adj_deps, 'adj')
