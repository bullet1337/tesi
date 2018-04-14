from collections import defaultdict
from typing import List, Set

from scripts.graph_building import generate_graph_par
from scripts.typez_and_constants import PosType, Token, Sentence, SynType, PrefixHandle, Edge, Node, Orientation
from scripts.utils import is_conjunctive, is_adversative


def is_adv(token: Token):
    return token.pos in {PosType.ADV, PosType.APRO}


def is_adv_adv_transition(edge: Edge, token: Token):
    return edge.type == SynType.сочин


def is_conj_adv_transition(edge: Edge, token: Token):
    return edge.type == SynType.соч_союзн


def is_adv_conj_transition(edge: Edge):
    return edge.token.pos == PosType.CONJ and edge.type == SynType.сочин \
           and (is_adversative(edge.token) or is_conjunctive(edge.token))


def is_adv_det_edge(edge: Edge):
    return edge.type == SynType.опред


def equal_adv(token1: Token, token2: Token):
    return True


def handle_adv_deps(node: Node, token: Token, sentence: Sentence):
    for edge in sentence.dep_tree[token]:
        if edge.orientation == Orientation.IN:
            continue

        if edge.type in {SynType.огранич, SynType.колич_огран, SynType.опред} \
            and edge.token.pos in {PosType.ADV, PosType.APRO}:
                token.ampl_token = edge.token
                node.process = False
                break


def generate_adv_graph_par(sentences: List[Sentence], sent_dict: defaultdict(set), exceptions: Set[str],
                           mode: PrefixHandle, strict_ee: bool):
    return generate_graph_par(sentences, sent_dict, exceptions, mode, strict_ee,
                              is_adv, is_adv_conj_transition, is_conj_adv_transition, is_adv_adv_transition,
                              is_adv_det_edge, equal_adv, handle_adv_deps, 'adv')
