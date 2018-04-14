from functools import reduce
from math import ceil
from typing import Set, List, Dict

from scripts.morpho_utils import convert_pm2_to_mystem
from scripts.typez_and_constants import Node, Prefix, MORPH, Token, PrefixHandle, Polarity, SynType, Orientation, \
    PosType, levels_prefixes, influence_table, PrefixCategory, Sentence, sgn
from scripts.utils import is_adversative, is_amplifier


def check_prefix(node: Node, token: Token, words: Set[str], prefix_cutted: bool, prefixes: List[Prefix]):
    analyzed_word = node.token_text if prefix_cutted else node.text

    for prefix in prefixes:
        if analyzed_word.startswith(prefix.value):
            without_prefix = analyzed_word[len(prefix.value):]
            if len(without_prefix) < 2:
                return None

            if without_prefix[0] in {'ы', 'ъ'} and prefix.value[-1] not in 'ёуеыаоэяию':
                without_prefix = ('и' if without_prefix[0] == 'ы' else '') + without_prefix[1:]

            tag = MORPH.tag(without_prefix)[0]
            if MORPH.word_is_known(without_prefix) \
                and convert_pm2_to_mystem(tag.POS, tag.grammemes - {tag.POS})[0] == token.pos \
                    or without_prefix in words:
                if prefix_cutted:
                    node.token_text = without_prefix
                else:
                    node.text = without_prefix
                return prefix
    return None


def handle_prefixes(node: Node, token: Token, mode: PrefixHandle, words: Set[str], exceptions: Set[str]):
    prefixes = [prefix for prefix in levels_prefixes[mode] if node.token_text.startswith(prefix.value)]
    prefix_cutted = node.text[:4] != node.token_text[:4] and prefixes
    if node.token_text not in exceptions and (prefix_cutted or node.text not in exceptions):
        first_prefix = check_prefix(node, token, words, prefix_cutted,
                                    prefixes if prefix_cutted else levels_prefixes[mode])
        if first_prefix:
            prefixes = [prefix for prefix in levels_prefixes[mode] if node.token_text.startswith(prefix.value)]
            prefix_cutted = node.text[:4] != node.token_text[:4] and prefixes
            if node.token_text not in exceptions and (node.text not in exceptions if not prefix_cutted else True):
                second_prefix = check_prefix(node, token, words, prefix_cutted,
                                             prefixes if prefix_cutted else levels_prefixes[mode])
                if second_prefix:
                    node.polarity = influence_table[(second_prefix.category, second_prefix.semantic)](node.polarity)
            elif node.token_text in exceptions:
                node.text = node.token_text
            node.polarity = influence_table[(first_prefix.category, first_prefix.semantic)](node.polarity)
    elif node.token_text in exceptions:
        node.text = node.token_text


def handle_invertion(token: Token, sentence: Sentence):
    for edge in sentence.dep_tree[token]:
        if edge.orientation == Orientation.IN:
            continue

        if edge.type == SynType.огранич:
            if edge.token.pos == PosType.PART and edge.token.text.lower() == 'не':
                token.context_tokens.append(edge.token)
                token.inverted = not token.inverted
            elif (edge.token.pos == PosType.ADV or edge.token.pos == PosType.CONJ) and is_adversative(edge.token):
                token.context_tokens.append(edge.token)
                token.inverted = not token.inverted


def init(token: Token):
    if 'ейш' in token.text.lower() or 'айш' in token.text.lower() or 'supr' in token.feats:
        return Polarity.VERY_POSITIVE
    return Polarity.SLIGHTLY_POSITIVE if 'еньк' in token.text.lower() or 'оньк' in token.text.lower() \
        else Polarity.POSITIVE


def get_polarity(node: Node, token: Token, sentence: Sentence, mode: PrefixHandle,
                 words: Set[str], exceptions: Set[str], deps_handler):
    node.polarity = init(token)

    handle_prefixes(node, token, mode, words, exceptions)
    handle_invertion(token, sentence)

    if 'adv' not in deps_handler.__name__ and token.inverted:
         node.polarity = influence_table[PrefixCategory.INVERTION](node.polarity)
    token.polarity = node.polarity

    deps_handler(node, token, sentence)


def avg_with_sgn(*polarities, sgn_function):
    polarities = [p for p in polarities if p != Polarity.NEUTRAL]
    if polarities:
        return Polarity(sgn_function(avg(polarities), polarities))
    else:
        return Polarity.NEUTRAL


def avg(args):
    return ceil(sum([abs(x.value) for x in args]) / len(args))


def any_sgn(value, args):
    return value * (-1 if (any([arg.value < 0 for arg in args])) else 1)


def all_sgn(value, args):
    return value * reduce(lambda x, y: x * sgn(y.value), args, 1)


def get_i_polarity(intensity: Token, polarity: Polarity, can_amplify: bool=True):
    if intensity is None:
        return polarity

    if can_amplify and is_amplifier(intensity):
        if intensity.polarity == Polarity.NEUTRAL:
            return polarity

        if (intensity.polarity.value < 0) ^ intensity.inverted:
            polarity = influence_table[(PrefixCategory.INVERTION, PrefixCategory.AMPLIFICATION)](polarity)
        else:
            polarity = influence_table[PrefixCategory.AMPLIFICATION](polarity)
    else:
        polarity = avg_with_sgn(intensity.polarity, polarity, sgn_function=any_sgn)
    return polarity


def get_p_polarity(polar: Polarity, polarity: Polarity):
    if polar is None:
        return polarity

    return avg_with_sgn(polar, polarity, sgn_function=any_sgn)


def get_s_polarity(subject: Polarity, polarity: Polarity):
    if subject is None:
        return polarity

    return avg_with_sgn(subject, polarity, sgn_function=all_sgn)


def get_o_polarity(object: Polarity, polarity: Polarity):
    if object is None:
        return polarity

    return avg_with_sgn(object, polarity, sgn_function=all_sgn)


def calc_polarity(meta: Dict[str, Token]):
    polarities = {}
    for k in ['p', 'po', 'ps', 'a']:
        if k in meta:
            polarities[k] = get_i_polarity(
                meta.get('ii' + k),
                get_i_polarity(meta.get('i' + k), meta[k].polarity, k != 'a')
            )

    if 's' in meta:
        polarities['s'] = get_p_polarity(polarities.get('ps'), meta['s'].polarity)
    if 'o' in meta:
        polarities['o'] = get_p_polarity(polarities.get('p'),
                                         get_p_polarity(polarities.get('po'),
                                                        get_s_polarity(polarities.get('s'), meta['o'].polarity)))
    if 'a' in polarities:
        polarities['a'] = get_p_polarity(polarities.get('p'), get_o_polarity(polarities.get('o'), polarities['a']))

    return polarities.get('a', polarities.get('o', polarities.get('p')))
