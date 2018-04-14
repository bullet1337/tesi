import copy
from collections import deque

from scripts.typez_and_constants import Sentence, State, Orientation
from scripts.utils import is_adversative


def process_state(edge_function, edge, token, state):
    if edge_function(edge, token):
        state.extended = True
        return True
    else:
        state.new_paths += 1
        return False


def get_paths_by_conjuctive(sentence: Sentence, node_filter, pos_conj_transition, conj_pos_transition,
                            pos_pos_transition):
    state = State()
    paths = []  # paths[path[group]]

    for token in sentence.dep_tree:
        if token.root:
            state.stack.append(token)
            if node_filter(token):
                state.states.append('A')
                state.paths.append([[]])
            else:
                state.states.append('U')

    current_state = None
    while state.stack:
        assert state.stack, str(sentence.id)
        token = state.stack[-1]
        if token in state.closed:
            token = state.stack.pop()
            temp_state = state.states.pop()
            if state.paths and state.paths[-1]:
                if state.paths[-1][-1] and token == state.paths[-1][-1][-1]:
                    state.paths[-1][-1].pop()
                    if not state.paths[-1][-1]:
                        state.paths[-1].pop()
                        if not state.paths[-1]:
                            state.paths.pop()
                elif is_adversative(token) and current_state != 'B' and temp_state == 'C':
                    state.paths[-1].pop()
                    if not state.paths[-1]:
                        state.paths.pop()
            continue

        assert state.states, str(sentence.id) + ' ' + str(token.id)
        current_state = state.states[-1]
        state.closed.add(token)
        if current_state != 'U':
            assert state.paths, str(sentence.id) + ' ' + str(token.id)
            if current_state == 'C':
                if is_adversative(token):
                    state.paths[-1].append([])
            else:
                state.paths[-1][-1].append(token)

        state.extended = False
        state.new_paths = 0
        new_tokens = deque()
        new_states = deque()
        for edge in sentence.dep_tree[token]:
            if edge.orientation == Orientation.IN:
                continue

            if current_state == 'U':
                new_tokens.append(edge.token)
                if node_filter(edge.token):
                    new_states.append('A')
                    state.paths.append([[]])
                else:
                    new_states.append('U')
            elif current_state == 'A' or current_state == 'B':
                if node_filter(edge.token):
                    if process_state(pos_pos_transition, edge, state.paths[-1][0][0], state):
                        new_tokens.appendleft(edge.token)
                        new_states.appendleft('B')
                    else:
                        new_tokens.append(edge.token)
                        new_states.append('A')
                elif pos_conj_transition(edge):
                    new_tokens.appendleft(edge.token)
                    new_states.appendleft('C')
                    state.extended = True
                else:
                    new_tokens.appendleft(edge.token)
                    new_states.appendleft('U')
            elif current_state == 'C':
                if node_filter(edge.token):
                    if process_state(conj_pos_transition, edge, state.paths[-1][0][0], state):
                        new_tokens.appendleft(edge.token)
                        new_states.appendleft('B')
                    else:
                        new_tokens.append(edge.token)
                        new_states.append('A')
                else:
                    new_tokens.appendleft(edge.token)
                    new_states.appendleft('U')

        for new_token, new_state in zip(new_tokens, new_states):
            state.stack.append(new_token)
            state.states.append(new_state)

        if not state.extended:
            if current_state != 'U':
                if current_state == 'B':
                    assert state.paths, str(sentence.id) + ' ' + str(token.id)
                    paths.append(copy.deepcopy(state.paths[-1]))
                elif current_state == 'C' and len(sum(state.paths[-1], [])) > 1:
                    if is_adversative(token):
                        state.paths[-1].pop()
                    assert state.paths, str(sentence.id) + ' ' + str(token.id)
                    paths.append(copy.deepcopy(state.paths[-1]))

        for i in range(state.new_paths):
            state.paths.append([[]])

    assert not state.stack, sentence.id
    assert not state.states, sentence.id
    assert len(state.closed) == len(sentence.word_ids), sentence.id
    state.closed.clear()
    assert not state.paths, sentence.id
    return paths


def get_paths_by_determinative(sentence: Sentence,
                               node_filter, edge_filter, pos_pos_filter):
    paths = []
    for token in sentence.dep_tree:
        #if token.pos in {PosType.S, PosType.SPRO, PosType.APRO, PosType.A, PosType.ANUM, PosType.V}:
        neighbors = []
        for edge in sentence.dep_tree[token]:
            if edge.orientation == Orientation.IN:
                continue

            if edge_filter(edge) and node_filter(edge.token) \
                    and (len(neighbors) == 0 or pos_pos_filter(edge.token, neighbors[0])):
                neighbors.append(edge.token)

        if len(neighbors) > 1:
            paths.append([neighbors])
    return paths