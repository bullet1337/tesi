from os import path
from subprocess import Popen, PIPE

from joblib import Parallel
from joblib import delayed
from nltk import PunktSentenceTokenizer

from scripts.typez_and_constants import Token, TokenType, NUM_CORES, Sentence, time_wrap


def segmentize_text(text):
    tokenizer = PunktSentenceTokenizer()
    sentences_bounds = tokenizer.span_tokenize(text, realign_boundaries=False)
    sentences = [Sentence(id=idx, left=bound[0], right=bound[1], text=text[bound[0]:bound[1]])
                 for idx, bound in enumerate(sentences_bounds)]
    return sentences


@time_wrap()
def segmentize(text_path):
    with open(text_path, mode='r', encoding='utf8') as split_file:
        text = split_file.read()
    return segmentize_text(text)


def tokenize(sentence):
    p = Popen(['scripts/greeb2tokenize.rb'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    greeb_output = p.communicate(input=sentence.text.encode(encoding='utf8'))[0].decode(encoding='utf8')

    text_position = 0
    for x in greeb_output.split('\n'):
        if x == '':
            break
        elem = x.split('\t')
        assert len(elem) == 4
        sentence.tokens.append(Token(id=len(sentence.tokens),
                                     left=int(elem[1]),
                                     right=int(elem[2]),
                                     text=elem[0],
                                     type=TokenType.from_string(elem[3])))
        if len(sentence.tokens) == 1:
            assert sentence.text.find(sentence.tokens[0].text) != -1
            text_position = sentence.text_left_bound + \
                sentence.text.find(sentence.tokens[0].text) - \
                sentence.tokens[0].text_left_bound
        sentence.tokens[-1].text_left_bound += text_position
        sentence.tokens[-1].text_right_bound += text_position

    return sentence


def tokenize_parallel(sentences, text_path):
    # Launch per sentence in case of large text
    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(tokenize)(sentence) for sentence in sentences)

    with open(text_path, mode='r', encoding='utf8') as split_file:
        text = split_file.read()
        for sentence in sentences:
            assert text[sentence.text_left_bound:sentence.text_right_bound] == sentence.text
            for token in sentence.tokens:
                assert text[token.text_left_bound:token.text_right_bound] == token.text


@time_wrap()
def tokenize_old(sentences, text_path):
    p = Popen(['scripts/greeb2tokenize.rb'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    greeb_output = p.communicate(input=' SENTENCE '.join([sentence.text for sentence in sentences])
                                 .encode(encoding='utf8'))[0].decode(encoding='utf8')

    s_idx = 0
    text_position = 0
    for x in greeb_output.split('\n'):
        if x == '':
            break

        elem = x.split('\t')
        if elem[0] == 'SENTENCE':
            text_position = 0
            s_idx += 1
            continue

        assert s_idx <= len(sentences)
        sentences[s_idx].tokens.append(Token(id=len(sentences[s_idx].tokens),
                                             left=int(elem[1]),
                                             right=int(elem[2]),
                                             text=elem[0],
                                             type=TokenType.from_string(elem[3])))
        if len(sentences[s_idx].tokens) == 1:
            assert sentences[s_idx].text.find(sentences[s_idx].tokens[0].text) != -1
            text_position = sentences[s_idx].text_left_bound + \
                sentences[s_idx].text.find(sentences[s_idx].tokens[0].text) - \
                sentences[s_idx].tokens[0].text_left_bound
        sentences[s_idx].tokens[-1].text_left_bound += text_position
        sentences[s_idx].tokens[-1].text_right_bound += text_position
        if sentences[s_idx].tokens[-1].type in {TokenType.LETTER, TokenType.INTEGER, TokenType.FLOAT}:
            sentences[s_idx].word_ids.add(sentences[s_idx].tokens[-1].id)

    if path.exists(text_path):
        with open(text_path, mode='r', encoding='utf8') as split_file:
            text = split_file.read()
    else:
        text = text_path
    for sentence in sentences:
        assert text[sentence.text_left_bound:sentence.text_right_bound] == sentence.text
        for token in sentence.tokens:
            assert text[token.text_left_bound:token.text_right_bound] == token.text

    return sentences


def normalize(sentence):
    sentence.word_ids = set()
    final_tokens = []
    i = 0
    while i < len(sentence.tokens):
        token = sentence.tokens[i]

        if token.id >= len(sentence.tokens) - 2:
            final_tokens.append(token)
            i += 1
        else:
            next_token = sentence.tokens[token.id + 1]
            after_next_token = sentence.tokens[token.id + 2]
            if token.type in {TokenType.LETTER, TokenType.INTEGER} \
                    and token.text_right_bound == next_token.text_left_bound \
                    and next_token.text_right_bound == after_next_token.text_left_bound \
                    and (next_token.text == '-' or next_token.text == '‐') \
                    and after_next_token.type == TokenType.LETTER:
                final_tokens.append(Token(id=len(final_tokens),
                                          left=token.text_left_bound,
                                          right=after_next_token.text_right_bound,
                                          text=token.text + next_token.text + after_next_token.text,
                                          type=TokenType.LETTER))
                i += 3
            else:
                final_tokens.append(token)
                i += 1
        token.id = len(final_tokens) - 1

        if token.type in {TokenType.LETTER, TokenType.INTEGER, TokenType.FLOAT}:
            sentence.word_ids.add(token.id)
    sentence.tokens = final_tokens
    return sentence


@time_wrap()
def normalize_parallel(sentences):
    sentences[:] = Parallel(n_jobs=NUM_CORES)(delayed(normalize)(sentence) for sentence in sentences)
