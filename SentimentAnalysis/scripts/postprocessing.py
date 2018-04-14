import itertools
from functools import reduce
from typing import Set, List

from joblib import Parallel
from joblib import delayed

from scripts.patterns import simple_patterns, cmp_patterns
from scripts.polarity import calc_polarity
from scripts.typez_and_constants import NUM_CORES, Polarity, time_wrap, sgn, Step, \
    Token, influence_table, PrefixSemantic, PrefixCategory, OR, AND, SimplePath, MatchingPath, Sentence, GROUP_RATIO


@time_wrap()
def set_tokens_polarity(graph):
    for node in graph:
        for sentence, token_entries in node.entries.items():
            for token_entry in token_entries:
                sentence.tokens[token_entry].polarity \
                    = Polarity(sgn(node.polarity.value) * sentence.tokens[token_entry].polarity.value)


def adapt_polarity(sentence: Sentence):
    processed = set()
    for token in sentence.tokens:
        if token not in processed and len(token.native_group) >= 2:
            polarities_dict = {True: [], False: []}
            for t in [token] + token.native_group:
                if t.polarity != Polarity.NEUTRAL:
                    polarities_dict[t.polarity.value > 0].append(t)
            if all(v for v in polarities_dict.values()):
                for b in [True, False]:
                    if len(polarities_dict[b]) / len(polarities_dict[not b]) >= GROUP_RATIO:
                        for t in polarities_dict[not b]:
                            t.polarity = Polarity(t.polarity.value - sgn(t.polarity.value))
                        break
            processed.update(token.native_group)
    return sentence


@time_wrap()
def adapt_polarity_par(sentences: List[Sentence]):
    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(adapt_polarity)(sentence) for sentence in sentences)


def process_node(token: Token, simple_path: SimplePath, sentence: Sentence, processed: Set[Token]=None,
                 previous: Token=None):
    init = MatchingPath([token])
    if simple_path.meta:
        init.meta.update({simple_path.meta: token})
    m_paths = [init]

    if processed is None:
        processed = set()
        if not token.entity:
            for e in token.group:
                init = MatchingPath([e])
                init.meta.update({simple_path.meta: e})
                m_paths.append(init)

    for step in simple_path.steps:
        temp =[]
        if isinstance(step, Step):
            for m_path in m_paths:
                extended = False
                for edge in sentence.dep_tree[m_path.path[-1]]:
                    if (previous is None or edge.token != previous) \
                            and (len(m_path.path) < 2 or edge.token != m_path.path[-2]) \
                            and step.check(edge):
                        for t in [edge.token] + edge.token.group:
                            if t in processed and (('comp' in t.feats) != ('comp' in edge.token.feats)):
                                continue

                            extended = True
                            processed.add(t)
                            temp_path = MatchingPath(m_path.path + [t])
                            temp_path.meta.update(m_path.meta)
                            if step.meta:
                                temp_path.meta.update({step.meta: t})
                            temp.append(temp_path)
                if not extended and step.optional:
                    temp.append(m_path)
        elif isinstance(step, SimplePath):
            for m_path in m_paths:
                next_simple_paths = process_node(m_path.path[-1], step, sentence, processed,
                                                m_path.path[-2] if len(m_path.path) > 1 else None)
                if next_simple_paths:
                    for next_simple_path in next_simple_paths:
                        temp_path = MatchingPath(m_path.path[:-1] + next_simple_path.path)
                        temp_path.meta.update(m_path.meta)
                        temp_path.meta.update(next_simple_path.meta)
                        temp.append(temp_path)
                elif step.optional:
                    temp.append(m_path)
        elif isinstance(step, OR):
            for m_path in m_paths:
                or_paths = []
                for or_simple_path in step.paths:
                    or_paths.extend(process_node(m_path.path[-1], or_simple_path, sentence, processed,
                                                 m_path.path[-2] if len(m_path.path) > 1 else None))
                if or_paths:
                    for or_path in or_paths:
                        temp_path = MatchingPath(m_path.path[:-1] + or_path.path)
                        temp_path.meta.update(m_path.meta)
                        temp_path.meta.update(or_path.meta)
                        temp.append(temp_path)
                elif step.optional:
                    temp.append(m_path)
        elif isinstance(step, AND):
            for m_path in m_paths:
                and_paths = []
                for and_simple_path in step.paths:
                    temp_res = process_node(m_path.path[-1], and_simple_path, sentence, processed,
                                            m_path.path[-2] if len(m_path.path) > 1 else None)
                    if temp_res:
                        temp_res = [res for res in temp_res if len(res.path) > 1]
                        if temp_res:
                            and_paths.append(temp_res)
                    elif not and_simple_path.optional:
                        and_paths = []
                        break
                if and_paths:
                    variants = itertools.product(*and_paths)
                    and_paths_variant = []
                    for variant in variants:
                        paths_sets = itertools.product(set(x.path) for x in variant)
                        if reduce(lambda x, y: x & (len(y) < 2 or len(y[0] & y[1]) == 0), paths_sets, True):
                            and_paths_variant.append(variant)

                    for and_paths in and_paths_variant:
                        temp_path = MatchingPath(m_path.path[:])
                        temp_path.meta.update(m_path.meta)
                        for and_path in and_paths:
                            temp_path.path.extend(and_path.path[1:])
                            temp_path.meta.update(and_path.meta)
                        temp.append(temp_path)
                elif step.optional:
                    temp.append(m_path)

        m_paths = temp
    return m_paths


def process_steps(sentence, pattern, is_cmp=False):
    entities = [token for token in sentence.tokens if token.entity]
    sentence_facts = []
    processed = set()
    for entity in entities:
        if entity in processed:
            continue

        m_paths = process_node(entity, pattern, sentence)

        if m_paths:
            temp_paths = []
            for m_path in m_paths:
                assert 'a' in m_path.meta or 'p' in m_path.meta or 'o' in m_path.meta, sentence.id
                assert is_cmp == ('ce' in m_path.meta), sentence.id
                m_path.meta['t'] = calc_polarity(m_path.meta)

                if 'pl' in m_path.meta.get('a', m_path.meta.get('o', m_path.meta.get('p'))).feats:
                    for e in m_path.meta['e'].group:
                        temp_path = MatchingPath([e] + m_path.path[1:])
                        temp_path.meta.update(m_path.meta)
                        temp_path.meta['e'] = e
                        temp_paths.append(temp_path)
            m_paths.extend(temp_paths)

            for m_path in m_paths:
                if not is_cmp and m_path.meta['e'].context_tokens:
                    m_path.meta['t'] \
                        = influence_table[(PrefixCategory.INVERTION, PrefixSemantic.MONO)](m_path.meta['t'])

                temp = []
                for t in m_path.path:
                    temp.extend(t.context_tokens)
                m_path.path.extend(temp)

                for k in ['p', 'po', 'ps', 'a']:
                    if k in m_path.meta:
                        m_path.meta[k + 'c'] = []
                        for kk in ['i', 'ii']:
                            if kk + k in m_path.meta:
                                m_path.meta[k + 'c'].append(m_path.meta[kk + k])
                                m_path.meta[k + 'c'].extend(m_path.meta[kk + k].context_tokens)
                m_path.path.sort(key=lambda token: token.id)

            sentence_facts.extend(m_paths)
            for fact in sentence_facts:
                print('%d: %s [%s]' % (sentence.id, fact.path, fact.meta['t']))

    return sentence_facts


def link_ta_to_ne(sentence, patterns, is_cmp=False):
    for pattern in patterns:
        sentence.facts.extend(process_steps(sentence, pattern, is_cmp))
    return sentence


@time_wrap()
def link_ta_to_ne_par(sentences):
    sentences[:] \
        = Parallel(n_jobs=NUM_CORES)(delayed(link_ta_to_ne)(sentence, simple_patterns) for sentence in sentences)
    sentences[:] \
        = Parallel(n_jobs=NUM_CORES)(delayed(link_ta_to_ne)(sentence, cmp_patterns, True) for sentence in sentences)
    print()
