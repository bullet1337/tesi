from subprocess import PIPE, Popen
from typing import List

from joblib import Parallel
from joblib import delayed

from scripts.postprocessing import process_steps, process_node
from scripts.typez_and_constants import BASE_DIR, time_wrap, Sentence, SynType, Orientation, NUM_CORES, State, \
    PosType, Step, Token, SimplePath, OR
from scripts.utils import is_conjunctive, is_adversative, is_comparative

FL_BIN = BASE_DIR + '/tools/freeling/bin/analyze'
FL_FLAGS = [
    '-f',
    BASE_DIR + '/configs/freeling/freeling.cfg'
]


@time_wrap()
def get_entities(sentences):
    text = '\n\n'.join(['\n'.join([sentence.tokens[id].text for id in sorted(sentence.word_ids)])
                                  for sentence in sentences]) + '\n\n'
    p = Popen([FL_BIN] + FL_FLAGS, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    fl_output = p.communicate(input=text.encode(encoding='utf8'))[0].decode(encoding='utf8').split('\n')

    fl_iter = iter(fl_output)
    for sentence in sentences:
        form, lemma, tag, _ = next(fl_iter).split()
        for token in sentence.tokens:
            if token.id not in sentence.word_ids:
                continue

            assert form.startswith(token.text), str(sentence.id) + ' ' + str(token.id)
            if tag == 'NP':
                token.entity = True
            form = form[len(token.text):]
            if form == '':
                temp = next(fl_iter).split()
                if not temp:
                    break
                form, lemma, tag, _ = temp
            else:
                assert form.startswith('_'), str(sentence.id) + ' ' + str(token.id)
                form = form[1:]
        assert form == '', sentence.id


def merge_entities(sentence: Sentence):
    entities = [token for token in sentence.dep_tree if token.entity]
    for entity in entities:
        if entity in sentence.dep_tree:
            edges = sentence.dep_tree[entity][:]
            while edges:
                edge = edges.pop()
                if edge.orientation == Orientation.OUT and edge.type in {SynType.аппоз, SynType.сочин}\
                        and edge.token.entity and abs(entity.id - edge.token.id) == 1:

                    sentence.dep_tree[entity].remove(edge)
                    new_edges = [edge for edge in sentence.dep_tree[edge.token] if edge.token != entity]
                    for new_edge in new_edges:
                        for change_edge in sentence.dep_tree[new_edge.token]:
                            if change_edge.token == edge.token:
                                change_edge.token = entity

                    sentence.dep_tree[entity].extend(new_edges)
                    tmp = sentence.dep_tree[entity]
                    edges.extend(new_edges)

                    sentence.dep_tree.pop(entity)
                    entity.text = entity.text + ' ' * (edge.token.text_left_bound - entity.text_right_bound) \
                                  + edge.token.text
                    entity.text_right_bound = edge.token.text_right_bound
                    sentence.dep_tree[entity] = tmp

                    sentence.dep_tree.pop(edge.token)

                    sentence.tokens.remove(edge.token)
                    sentence.word_ids.remove(edge.token.id)
                    for token in sentence.tokens[edge.token.id:]:
                        if token.id in sentence.word_ids:
                            sentence.word_ids.remove(token.id)
                            sentence.word_ids.add(token.id - 1)
                        token.id -= 1
    return sentence


def get_enitities_context(sentence: Sentence):
    entities = [token for token in sentence.dep_tree if token.entity]
    for entity in entities:
        for edge in sentence.dep_tree[entity]:
            if edge.orientation == Orientation.IN:
                continue

            if edge.type == SynType.огранич:
                if edge.token.pos == PosType.PART and edge.token.text.lower() == 'не':
                    entity.context_tokens.append(edge.token)
    return sentence


def get_ce_by_regexp(sentence, state: State):
    paths = []

    for token in sentence.dep_tree:
        if token.root:
            state.stack.append(token)
            if token.entity:
                state.states.append('A')
                state.paths.append([])
            else:
                state.states.append('U')

    while state.stack:
        assert state.stack, str(sentence.id)
        token = state.stack[-1]
        if token in state.closed:
            token = state.stack.pop()
            state.states.pop()
            if state.paths and state.paths[-1]:
                if state.paths[-1] and token == state.paths[-1][-1]:
                    state.paths[-1].pop()
                    if not state.paths[-1]:
                        state.paths.pop()
            continue

        assert state.states, str(sentence.id) + ' ' + str(token.id)
        current_state = state.states[-1]
        state.closed.add(token)
        if current_state != 'U':
            assert state.paths, str(sentence.id) + ' ' + str(token.id)
            if current_state != 'C':
                state.paths[-1].append(token)

        state.extended = False
        state.new_paths = 0
        for edge in sentence.dep_tree[token]:
            if edge.orientation == Orientation.IN:
                continue

            state.stack.append(edge.token)
            if current_state == 'U':
                if edge.token.entity:
                    state.states.append('A')
                    state.paths.append([])
                else:
                    state.states.append('U')
            elif current_state == 'A' or current_state == 'B':
                if edge.token.entity:
                    if edge.type in {SynType.аппоз, SynType.примыкат, SynType.сочин, SynType.об_аппоз}:
                        state.states.append('B')
                        state.extended = True
                    else:
                        state.states.append('A')
                        state.new_paths += 1
                        state.extended = False
                elif edge.token.pos == PosType.CONJ and edge.type in {SynType.сочин, SynType.сравнит}\
                        and is_conjunctive(edge.token):
                    state.states.append('C')
                    state.extended = True
                else:
                    state.states.append('U')
            elif current_state == 'C':
                if edge.token.entity:
                    if edge.type in {SynType.соч_союзн, SynType.сравн_союзн}:
                        state.states.append('B')
                        state.extended = True
                    else:
                        state.states.append('A')
                        state.new_paths += 1
                        state.extended = False
                else:
                    state.states.append('U')

        if not state.extended:
            if current_state != 'U':
                if current_state == 'B':
                    assert state.paths, str(sentence.id) + ' ' + str(token.id)
                    paths.append(state.paths[-1][:])
                elif current_state == 'C' and len(state.paths[-1]) > 1:
                    assert state.paths, str(sentence.id) + ' ' + str(token.id)
                    paths.append(state.paths[-1][:])

        for i in range(state.new_paths):
            state.paths.append([])

    assert not state.stack, sentence.id
    assert not state.states, sentence.id
    assert len(state.closed) == len(sentence.word_ids), sentence.id
    state.closed.clear()
    assert not state.paths, sentence.id

    return [set(path) for path in paths]


def get_ce_by_patterns(sentence, patterns):
    groups = []
    for pattern in patterns:
        groups.extend(process_steps(sentence, pattern))
    return groups


def check2(token: Token):
    return token.pos == PosType.CONJ and (is_conjunctive(token) or is_adversative(token) or is_comparative(token))


def check3(token: Token):
    return token.entity


def check4(token: Token):
    return token.pos in {PosType.S, PosType.SPRO}


def check5(token: Token):
    return token.pos in {PosType.S, PosType.SPRO, PosType.V, PosType.A} and 'comp' not in token.feats


def get_conjuctive_entities(sentence):
    get_conjuctive_entities.state = State()
    get_conjuctive_entities.patterns = [
        SimplePath(steps=[
            OR(paths=[
                SimplePath(steps=[
                    Step(
                        orientations={Orientation.OUT},
                        edge_types={SynType.аппоз, SynType.об_аппоз, SynType.ном_аппоз},
                        token_check=check4
                    )
                ]),
                SimplePath(steps=[
                    Step(
                        orientations={Orientation.IN},
                        edge_types={SynType.предик},
                        token_check=check5
                    )
                ])
            ]),
            Step(
                orientations={Orientation.OUT},
                edge_types={SynType.сочин, SynType.сравнит},
                token_check=check2
            ),
            Step(
                orientations={Orientation.OUT},
                edge_types={SynType.соч_союзн, SynType.сравн_союзн},
                token_check=check3,
                meta='e2'
            )
        ], meta='e1')
    ]

    groups1 = get_ce_by_regexp(sentence, get_conjuctive_entities.state)
    groups2 = []
    entities = [token for token in sentence.tokens if token.entity]
    for entity in entities:
        for pattern in get_conjuctive_entities.patterns:
            groups2.extend(process_node(entity, pattern, sentence))

    for group2 in groups2:
        first, second = group2.meta['e1'], group2.meta['e2']
        first_g = second_g = None
        for group1 in groups1:
            if first in group1:
                first_g = group1
            if second in group1:
                second_g = group1

        if first_g is None and second_g is None:
            groups1.append({first, second})
        elif first_g != second_g:
            if first_g is None:
                second_g.add(first)
            elif second_g is None:
                first_g.add(second)
            elif first_g or second_g:
                first_g.update(second_g)
                groups1.remove(second_g)

    for group in groups1:
        for token in group:
            token.group.extend(t for t in group if t != token)
    return sentence


@time_wrap()
def merge_entities_par(sentences: List[Sentence]):
    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(merge_entities)(sentence) for sentence in sentences)
    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(get_enitities_context)(sentence) for sentence in sentences)
    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(get_conjuctive_entities)(sentence) for sentence in sentences)
