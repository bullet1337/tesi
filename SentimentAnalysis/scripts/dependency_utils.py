from collections import defaultdict
from subprocess import PIPE, Popen

from joblib import Parallel
from joblib import delayed

from scripts.typez_and_constants import Sentence, Edge, SynType, Orientation, BASE_DIR, NUM_CORES, time_wrap

MP_BIN = 'java'
MP_CONFIG = 'MPSTR_mystem.mco'
MP_JAR = BASE_DIR + '/tools/dependency_parser/maltparser/maltparser-1.9.0.jar'
MP_FLAGS = ['-Xmx8g', '-jar', MP_JAR, '-c', MP_CONFIG, '-m', 'parse']


def get_dep_tree(sentence: Sentence, dep_tree):
    sentence.dep_tree = defaultdict(list)
    assert len(dep_tree) == len(sentence.word_ids)
    words_list = sorted(sentence.word_ids)
    for dependency in dep_tree:
        to_token = sentence.tokens[words_list[int(dependency[0]) - 1]]
        if int(dependency[1]) > 0:
            from_token = sentence.tokens[words_list[int(dependency[1]) - 1]]
            sentence.dep_tree[from_token].append(Edge(SynType.from_string(dependency[2]), to_token, Orientation.OUT))
            sentence.dep_tree[to_token].append(Edge(SynType.from_string(dependency[2]), from_token, Orientation.IN))
        else:
            sentence.dep_tree[to_token]
            to_token.root = True
    return sentence


@time_wrap()
def get_dep_tree_parallel(sentences):
    # Only ONE launch on whole text
    mp_input = ''
    for sentence in sentences:
        for id in sorted(sentence.word_ids):
            token = sentence.tokens[id]
            mp_input += '\t'.join([str(token.id), token.text, token.lemma, token.pos.name, token.pos.name,
                                   '|'.join(token.feats) or '_'])
            mp_input += '\n'
            id += 1
        mp_input += '\n'

    p = Popen([MP_BIN] + MP_FLAGS, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    mp_output = p.communicate(input=mp_input.encode(encoding='utf8'))[0].decode(encoding='utf8')

    sentences_dep_trees = [[[x.split('\t')[0], x.split('\t')[6], x.split('\t')[7]] for x in tree.split('\n')]
                           for tree in list(filter(None, mp_output.split('\n\n')))]
    assert len(sentences_dep_trees) == len(sentences)

    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(get_dep_tree)(sentences[i], sentences_dep_trees[i])
                                              for i in range(len(sentences)))
