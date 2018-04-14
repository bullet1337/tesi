from collections import defaultdict
from typing import List, Set

from scripts.graph_building import generate_graph_par
from scripts.patterns import noun_check, adj_not_comp_check
from scripts.typez_and_constants import PosType, Token, Sentence, SynType, PrefixHandle, Edge, \
    interesting_feats, Node, Orientation, Polarity
from scripts.utils import is_conjunctive, is_adversative


def is_noun(token: Token):
    return token.pos == PosType.S and not token.entity


def is_noun_noun_transition(edge: Edge, token: Token):
    return edge.type in {SynType.аппоз, SynType.примыкат, SynType.сочин, SynType.об_аппоз} \
           and equal_noun(edge.token, token)


def is_conj_noun_transition(edge: Edge, token: Token):
    return edge.type == SynType.соч_союзн and equal_noun(edge.token, token)


def is_noun_conj_transition(edge: Edge):
    return edge.token.pos == PosType.CONJ and edge.type == SynType.сочин \
           and (is_adversative(edge.token) or is_conjunctive(edge.token))


def is_noun_det_edge(edge: Edge):
    return edge.type in SynType.опред


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


def equal_noun(token1: Token, token2: Token):
    return get_token_info(token1)['quantity'] == get_token_info(token2)['quantity']


def handle_noun_deps(node: Node, token: Token, sentence: Sentence):
    for edge in sentence.dep_tree[token]:
        if edge.orientation == Orientation.IN:
            continue

        if (edge.type in {SynType._1_компл, SynType._2_компл, SynType._3_компл, SynType._4_компл, SynType._5_компл,
                          SynType.аппоз, SynType.об_аппоз, SynType.ном_аппоз} and noun_check(edge.token)
                or edge.type in {SynType.опред, SynType.оп_опред} and adj_not_comp_check(edge.token)) \
                and edge.token.polarity != Polarity.NEUTRAL:
            node.process = False
            break


def generate_noun_graph_par(sentences: List[Sentence], sent_dict: defaultdict(set), exceptions: Set[str],
                            mode: PrefixHandle, strict_ee: bool):
    return generate_graph_par(sentences, sent_dict, exceptions, mode, strict_ee,
                              is_noun, is_noun_conj_transition, is_conj_noun_transition, is_noun_noun_transition,
                              None, None, handle_noun_deps, 'noun')
