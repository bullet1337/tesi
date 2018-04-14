from collections import defaultdict
from typing import List, Set

from scripts.graph_building import generate_graph_par
from scripts.patterns import noun_check, adj_not_comp_check, adv_check
from scripts.typez_and_constants import PosType, Token, Sentence, SynType, PrefixHandle, Edge, \
    interesting_feats, Node, Orientation, Polarity
from scripts.utils import is_conjunctive, is_adversative


def is_verb(token: Token):
    return token.pos == PosType.V and 'partcp' not in token.feats and not token.entity


def is_verb_verb_transition(edge: Edge, token: Token):
    return edge.type in {SynType.примыкат, SynType.сочин} \
           and equal_verb(edge.token, token)


def is_conj_verb_transition(edge: Edge, token: Token):
    return edge.type == SynType.соч_союзн and equal_verb(edge.token, token)


def is_verb_conj_transition(edge: Edge):
    return edge.token.pos == PosType.CONJ and edge.type == SynType.сочин \
           and (is_adversative(edge.token) or is_conjunctive(edge.token))


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


def equal_verb(token1: Token, token2: Token):
    return get_token_info(token1)['quantity'] == get_token_info(token2)['quantity']


def handle_verb_deps(node: Node, token: Token, sentence: Sentence):
    for edge in sentence.dep_tree[token]:
        if edge.orientation == Orientation.IN:
            continue

        if (edge.type in {SynType._1_компл, SynType.обст_тавт, SynType.об_обст, SynType.суб_обст, SynType.обст}
                and noun_check(edge.token)
                or edge.type in {SynType.огранич, SynType.колич_огран, SynType.опред, SynType.обст}
                and adv_check(edge.token) or edge.type in {SynType.присвяз} and adj_not_comp_check(edge.token)) \
                and edge.token.polarity != Polarity.NEUTRAL:
            node.process = False
            break


def generate_verb_graph_par(sentences: List[Sentence], sent_dict: defaultdict(set), exceptions: Set[str],
                            mode: PrefixHandle, strict_ee: bool):
    return generate_graph_par(sentences, sent_dict, exceptions, mode, strict_ee,
                              is_verb, is_verb_conj_transition, is_conj_verb_transition, is_verb_verb_transition,
                              None, None, handle_verb_deps, 'verb')
