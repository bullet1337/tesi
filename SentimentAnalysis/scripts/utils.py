import os
from collections import defaultdict
from typing import List

import jinja2
from graphviz import Digraph

from scripts.typez_and_constants import Orientation, PosType, Token, Polarity, adversative, conjunctive, comparative, \
    amplifiers, Sentence


def is_adj_or_part(token: Token):
    return not token.entity and (token.pos == PosType.A or (token.pos == PosType.V and 'partcp' in token.feats))


def is_part(token: Token):
    return not token.entity and (token.pos == PosType.V and 'partcp' in token.feats)


def is_comparative(token: Token):
    return token.text.lower() in comparative


def is_conjunctive(token: Token):
    return token.text.lower() in conjunctive


def is_adversative(token: Token):
    return token.text.lower() in adversative


def is_amplifier(token: Token):
    return token.text.lower() in amplifiers


def print_depenency_tree(sentence, dir):
    g = Digraph('SENTENCE_' + str(sentence.id), directory=dir, format='pdf', encoding='utf8',
                graph_attr={'label': sentence.text}, node_attr={'style': 'filled'})
    sentence.pdf_path = g.filepath + '.pdf'
    for id in sentence.word_ids:
        token = sentence.tokens[id]
        g.node(name=str(token.id), label=token.text + '\n' + token.pos.name,
               fillcolor='red' if token.polarity.value < 0 else 'green' if token.polarity.value > 0 else 'lightgray',
               color='blue' if token.entity else '',
               penwidth='3' if token.entity else '1',
               shape='octagon' if token.tonal_facts else 'box' if token.entity else 'ellipse')
        """
        g.node(name=str(token.id), label=token.text + '\n' + token.pos.name,
               fillcolor='green' if is_adj_or_part(token) else 'lightgray')
        """
    for token, edge_list in sentence.dep_tree.items():
        for edge in edge_list:
            if edge.orientation == Orientation.OUT and token != edge.token:
                g.edge(str(token.id), str(edge.token.id), label=edge.type and edge.type.name)
    g.render()


def print_sentiment_graph(graph, dir, filename, with_single=True):
    g = Digraph('SENTIMENT_GRAPH', directory=dir, format='pdf', encoding='utf8', filename=filename)
    g.body = ['rankdir=LR']
    for node, edges in graph.items():
        if not with_single and len(graph[node]) == 0:
            continue

        color = ''
        if node.polarity.value > 0:
            color = 'green'
        elif node.polarity.value != 0:
            color = 'red'
        g.node(name=node.text, label=node.text, style='filled', fillcolor=color)
        for edge in edges:
            edge_color = ''
            if len(edge.polarities[True]) > len(edge.polarities[False]):
                edge_color = 'green'
            elif len(edge.polarities[True]) < len(edge.polarities[False]):
                edge_color = 'red'
            g.edge(node.text, edge.node.text,
                   label='<<font color="green">%s</font><font color="red">%s</font>W=%d>'
                             % (str([(pair[0].value, pair[1].value) for pair in edge.polarities[True]]),
                                str([(pair[0].value, pair[1].value) for pair in edge.polarities[False]]),
                                edge.weight),
                   color=edge_color)
    g.render()
    print('SENTIMENT GRAPH GENERATED AT: %s' % g.filepath)


def render(tmpl_path, context):
    path, filename = os.path.split(tmpl_path)
    return jinja2.Environment(loader=jinja2.FileSystemLoader(path or './')).get_template(filename).render(context)


def get_dict(filepath, strict_ee):
    sent_dict = {}
    exceptions = set()
    with open(filepath, encoding='utf8', mode='r') as file:
        for line in file:
            info = line.strip('\n').split('\t')
            w, wp, hw, hwp, is_ampl = info + [None] * (5 - len(info))
            if not strict_ee:
                w = w.replace('ё', 'е')
                if hw:
                    hw = hw.replace('ё', 'е')

            if wp:
                sent_dict[w] = Polarity.from_string(wp.replace(' ', '_'))
            if hwp:
                sent_dict[hw] = Polarity.from_string(hwp.replace(' ', '_'))

            if hw and (not wp or not hwp):
                exceptions.add(w if len(w) > len(hw) else hw)

            if is_ampl:
                amplifiers.add(hw or w)
    for x in sent_dict:
        assert sent_dict[x], x

    # text = ' . '.join(sent_dict.keys())
    # words_lemmas = lemmatize_and_tag_parallel(tokenize_old(segmentize_text(text), text))
    # assert len(words_lemmas) == len(sent_dict)
    # words_lemmas = [x.tokens[0].lemma if is_part(x.tokens[0]) else x.tokens[0].text for x in words_lemmas]
    # sent_dict = {k: v for k, v in zip(words_lemmas, sent_dict.values())}

    # text = ' . '.join(exceptions)
    # words_lemmas = lemmatize_and_tag_parallel(tokenize_old(segmentize_text(text), text))
    # assert len(words_lemmas) == len(exceptions)
    # exceptions = {x.tokens[0].lemma if is_part(x.tokens[0]) else x.tokens[0].text for x in words_lemmas}
    print('%s DICT BUILDED: %d POS AND %d NEG WORDS (%d TOTAL), %d EXCEPTIONS' % (
            os.path.basename(filepath).split('.')[0].upper(),
            len([x for x in sent_dict.values() if x.value > 0]),
            len([x for x in sent_dict.values() if x.value < 0]),
            len(sent_dict),
            len(exceptions)
        )
    )
    return sent_dict, exceptions


def get_pos_stats(sentences: List[Sentence], words_count):
    stats = defaultdict(int)
    for sentence in sentences:
        for token in sentence.tokens:
            if token.pos in {PosType.S, PosType.CONJ}:
                stats[token.pos.name] += 1
            elif token.pos in {PosType.ADV, PosType.APRO}:
                stats['ADV'] += 1
            elif token.pos == PosType.A or (token.pos == PosType.V and 'partcp' in token.feats):
                stats['A'] += 1
            elif token.pos == PosType.V and 'partcp' not in token.feats:
                stats['V'] += 1
    return {k: round(v * 100 / words_count) for k, v in stats.items()}