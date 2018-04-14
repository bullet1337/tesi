import json
import re
from itertools import groupby
from subprocess import Popen, PIPE

from joblib import Parallel
from joblib import delayed

from scripts.typez_and_constants import BASE_DIR, transformations, PosType, Token, Sentence, NUM_CORES, TokenType, MORPH, \
    time_wrap

MYSTEM_BIN = BASE_DIR + '/tools/lemmatizer/mystem'
MYSTEM_FLAGS = [
    '-c',
    '-n',
    '-d',
    '-i',
    '--format',
    'json',
    '--eng-gr'
]


def get_next_word(json_output, idx, form):
    data = json.loads(json_output[idx])
    text = data['text'].strip()
    if text == '':
        return '', '', ''

    feats = ''
    lemma = ''
    list_text = text.split()
    for txt in list_text[1:]:
        data['text'] = txt
        json_output.insert(idx + 1, json.dumps(data))
    text = list_text[0]
    if 'analysis' not in data:
        form += text
        lemma += MORPH.parse(text)[0].normal_form
    else:
        form += text
        if len(data['analysis']) > 0:
            lemma += data['analysis'][0]['lex']
            feats = data['analysis'][0]['gr']

    return form, lemma, feats


def convert_pm2_to_mystem(pos: str, feats: list):
    pos_transform = transformations[pos]
    if type(pos_transform) is dict:
        found = False
        for key in pos_transform.keys():
            if key and key in feats:
                pos_transform = pos_transform[key]
                found = True
        if not found:
            pos_transform = pos_transform[None]
    pos_transform = pos_transform.split(',')

    feats_transform = set()
    for feat in feats:
        feat_transform = transformations.get(feat)
        if feat_transform is None:
            # print(feat)
            continue

        if type(feat_transform) is dict:
            found = False
            for key in feat_transform.keys():
                if key and key in feats:
                    feat_transform = feat_transform[key]
                    found = True
            if not found:
                feat_transform = feat_transform[None]
        feats_transform.add(feat_transform)

    return PosType.from_string(pos_transform[0]), set(pos_transform[1:]) | feats_transform


def assign_lemma_and_pos(token: Token, lemma: str, feats: str):
    if lemma != '':
        token.lemma = lemma
    else:
        print(token.text)
        token.lemma = MORPH.parse(token.text)[0].normal_form
    if feats != '':
        feats = re.split('[=(,]', feats.split('|')[0])
        token.pos = PosType.from_string(feats[0])
        if token.pos == PosType.ADV and 'comp' in feats:
            p = MORPH.parse(token.text)[0]
            if p.tag._POS == 'COMP' and 'Qual' in p.tag.grammemes:
                token.pos, token.feats = convert_pm2_to_mystem(p.tag._POS, p.tag.grammemes - {p.tag._POS})
                token.lemma = p.normal_form
            else:
                print(token.text)
                token.feats = set(filter(None, feats[1:]))
        else:
            token.feats = set(filter(None, feats[1:]))
    else:
        tag = MORPH.tag(token.text)[0]
        token.pos, token.feats = convert_pm2_to_mystem(tag._POS, tag.grammemes - {tag._POS})

    if 'оньк' in token.lemma or 'еньк' in token.lemma:
        strict = token.lemma.replace('оньк', '').replace('еньк', '')
        p = MORPH.parse(strict)[0]
        pos, _ = convert_pm2_to_mystem(p.tag._POS, p.tag.grammemes - {p.tag._POS})
        if pos == token.pos:
            token.lemma = strict


def lemmatize_and_tag(sentence: Sentence, info):
    i = 0
    form = lemma = feats = ''
    for token in sentence.tokens:
        if token.id not in sentence.word_ids:
            token.lemma = MORPH.parse(token.text)[0].normal_form
            continue

        while not token.text.startswith(form) or form == '':
            assert i < len(info), token.text + ' ' + str(sentence.id)
            form = ''
            form, lemma, feats = get_next_word(info, i, form)
            i += 1

        if token.text != form:
            while form != token.text:
                assert i < len(info), token.text
                form, lemma, feats = get_next_word(info, i, form)
                if form in token.text:
                    i += 1
                else:
                    break
            lemma = ''
        form = ''
        assign_lemma_and_pos(token, lemma, feats)

    return sentence


@time_wrap()
def lemmatize_and_tag_parallel(sentences):
    # Only ONE launch on whole text
    text = ' . SENTENCE . '.join([' '.join([sentence.tokens[id].text for id in sorted(sentence.word_ids)])
                                  for sentence in sentences])
    p = Popen([MYSTEM_BIN] + MYSTEM_FLAGS, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    mystem_output = p.communicate(input=text.encode(encoding='utf8'))[0].decode(encoding='utf8')
    results = mystem_output.split('\n')
    sentences_info = [list(g) for k, g in groupby(results, lambda x: 'SENTENCE' not in x) if k]
    assert len(sentences_info), len(sentences)

    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(lemmatize_and_tag)(sentences[i], sentences_info[i])
                                              for i in range(len(sentences)))

    for sentence in sentences:
        for token in sentence.tokens:
            assert token.type != TokenType.LETTER and token.type != TokenType.INTEGER and token.type != TokenType.FLOAT\
                   or token.lemma is not None
    return sentences
