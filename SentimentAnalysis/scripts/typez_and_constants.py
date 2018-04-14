import multiprocessing
import os
import time
from collections import OrderedDict
from collections import defaultdict
from enum import Enum
from typing import List, Set

import pymorphy2

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/..'
NUM_CORES = multiprocessing.cpu_count()
MORPH = pymorphy2.MorphAnalyzer()
GROUP_RATIO = 3 / 2
CATEGORIES = OrderedDict([
    ('society', 'Общество'),
    ('religion', 'Религия'),
    ('culture', 'Культура'),
    ('films', 'Рецензии'),
    ('test', 'Тест'),
])

transformations = {
    'UNKN': 'UNKN',
    'VERB': 'V',
    'PRTF': 'V,partcp,brev',
    'PRTS': 'V,partcp,plen',
    'GRND': 'V,ger',
    'INFN': 'V,inf',
    'NOUN': 'S',
    'NPRO': 'SPRO',
    'LATN': 'S',
    'ADVB': {None: 'ADV', 'Dmns': 'ADVPRO', 'Ques': 'ADVPRO'},
    'PREP': 'PR',
    'PRED': 'ADV,praed',
    'NUMR': 'NUM',
    'NUMB': 'NUM',
    'ROMN': 'NUM',
    'CONJ': 'CONJ',
    'PRCL': 'PART',
    'INTJ': 'INTJ',
    'PNCT': 'UNKN',
    'COMP': 'A',
    'ADJF': {None: 'A,plen', 'Anum': 'ANUM', 'Apro': 'APRO'},
    'ADJS': {None: 'A,brev', 'Anum': 'ANUM', 'Apro': 'APRO'},
    #
    'Qual': {None: 'comp', 'Supr': 'supr'},
    #
    'anim': 'anim',
    'inan': 'inan',
    #
    'masc': 'm',
    'femn': 'f',
    'neut': 'n',
    'ms-f': 'mf',
    'Ms-f': 'mf',
    #
    'sing': 'sg',
    'plur': 'pl',
    #
    'nomn': 'nom',
    'gent': 'gen',
    'datv': 'dat',
    'accs': 'acc',
    'ablt': 'ins',
    'loct': 'abl',
    'voct': 'voc',
    'gen1': 'gen',
    'gen2': 'part',
    'acc2': 'acc',
    'loc1': 'loc',
    'loc2': 'loc',
    #
    'perf': 'pf',
    'impf': 'ipf',
    #
    'tran': 'tran',
    'intr': 'intr',
    #
    '1per': '1p',
    '2per': '2p',
    '3per': '3p',
    #
    'pres': 'praes',
    'past': 'praet',
    'futr': 'inpraes',
    #
    'indc': 'indic',
    'impr': 'imper',
    #
    'actv': 'act',
    'pssv': 'pass',
    #
    'Poss': 'poss',
    'Prnt': 'parenth',
    'Dist': 'dist',
    'Slng': 'obsc',
    'V-ie': 'patrn',
    'Patr': 'patrn',
    'Infr': 'inform',
    'Abbr': 'abbr',
    'Arch': 'obsol',
    'Surn': 'famn',
    'Name': 'persn',
}

amplifiers = set()

# да == и, но
conjunctive = {
    'и',
    'а',
    'также',
    'тоже',
    'ни',
}

adversative = {
    'но',
    'зато',
    'однако',
    'хоть',
    'хотя'
}

comparative = {
    'как',
    'чем'
}

interesting_feats = {
    'gender': {'m', 'f', 'n', 'mf'},
    'quantity': {'sg', 'pl'},
    'case': {'nom', 'gen', 'dat', 'acc', 'ins', 'abl', 'part', 'loc', 'voc'}
}


class AutoEnum(Enum):
    def __new__(cls, value=None):
        value = value if value is not None else len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @classmethod
    def from_string(cls, val: str):
        return getattr(cls, val.upper(), None)


class PrefixCategory(AutoEnum):
    INVERTION = ()
    AMPLIFICATION = ()


class PrefixSemantic(AutoEnum):
    MONO = ()
    POLY = ()


class Prefix:
    def __init__(self, value, category, semantic):
        self.value = value
        self.category = category
        self.semantic = semantic

prefixes = [
    Prefix('а', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('ан', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('анти', PrefixCategory.INVERTION, PrefixSemantic.MONO),
    Prefix('архи', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('без', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('бес', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('гипер', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('де', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('дез', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('диз', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('дис', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('им', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('ин', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('ир', PrefixCategory.INVERTION, PrefixSemantic.POLY),
    Prefix('мега', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('наи', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('не', PrefixCategory.INVERTION, PrefixSemantic.MONO),
    Prefix('пре', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('сверх', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('супер', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
    Prefix('ультра', PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO),
]


class PrefixHandle(AutoEnum):
    WEAK = ()
    MEDIUM = ()
    STRICT = ()


levels_prefixes = {
    PrefixHandle.WEAK: {prefix for prefix in prefixes},
    PrefixHandle.MEDIUM: {prefix for prefix in prefixes if prefix.semantic == PrefixSemantic.MONO},
    PrefixHandle.STRICT: {}
}


class SynType(AutoEnum):
    # Актантные СинтО
    предик = ()
    дат_субъект = ()
    агент = ()
    квазиагент = ()
    несобст_агент = ()
    _1_компл = ()
    _2_компл = ()
    _3_компл = ()
    _4_компл = ()
    _5_компл = ()
    присвяз = ()
    _1_несобст_компл = ()
    _2_несобст_компл = ()
    _3_несобст_компл = ()
    _4_несобст_компл = ()
    неакт_компл = ()
    компл_аппоз = ()
    предл = ()
    подч_союзн = ()
    инф_союзн = ()
    сравнит = ()
    сравн_союзн = ()
    электив = ()
    сент_предик = ()
    адр_присв = ()
    # Определительные СинтО
    опред = ()
    оп_опред = ()
    аппрокс_порядк = ()
    релят = ()
    # Общеатрибутивные СинтО
    атриб = ()
    композ = ()
    # Аппозитивные СинтО
    аппоз = ()
    об_аппоз = ()
    ном_аппоз = ()
    нум_аппоз = ()
    # Количественные СинтО
    количест = ()
    аппрокс_колич = ()
    колич_копред = ()
    колич_огран = ()
    распред = ()
    аддит = ()
    # Обстоятельственные СинтО
    обст = ()
    длительн = ()
    кратно_длительн = ()
    дистанц = ()
    обст_тавт = ()
    суб_обст = ()
    об_обст = ()
    суб_копр = ()
    об_копр = ()
    огранич = ()
    вводн = ()
    изъясн = ()
    разъяснит = ()
    примыкат = ()
    уточн = ()
    # Сочинительные СинтО
    сочин = ()
    сент_соч = ()
    соч_союзн = ()
    кратн = ()
    # Служебные СинтО
    аналит = ()
    пассивно_аналитическое = ()
    вспомогательное = ()
    колич_вспом = ()
    соотнос = ()
    эксплет = ()
    пролепт = ()

    @classmethod
    def from_string(cls, val: str):
        return getattr(cls, ('_' if val[0].isdigit() else '') + val.replace(' ', '').replace('-', '_'), None)


class TokenType(AutoEnum):
    LETTER = ()
    FLOAT = ()
    INTEGER = ()
    SPUNCT = ()
    PUNCT = ()
    SEPAR = ()
    BREAK = ()
    RESIDUAL = ()

    @classmethod
    def from_string(cls, val: str):
        return getattr(cls, val.upper(), None)


class Orientation(AutoEnum):
    IN = ()
    OUT = ()


class Polarity(AutoEnum):
    VERY_POSITIVE = 3
    POSITIVE = 2
    SLIGHTLY_POSITIVE = 1
    NEUTRAL = 0
    SLIGHTLY_NEGATIVE = -1
    NEGATIVE = -2
    VERY_NEGATIVE = -3

    def __bool__(self):
        return self.value != 0


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def assert_not_neutral(x):
    assert x != Polarity.NEUTRAL, x
    return True

influence_table = {
    # ненадежный
    (PrefixCategory.INVERTION, PrefixSemantic.MONO):
        lambda x: assert_not_neutral and Polarity(-x.value),
    # безнадежный
    (PrefixCategory.INVERTION, PrefixSemantic.POLY):
        lambda x: assert_not_neutral and Polarity(-x.value if abs(x.value) == 1 else -x.value + sgn(x.value)),
    # пренадежный
    (PrefixCategory.AMPLIFICATION, PrefixSemantic.MONO):
        lambda x: assert_not_neutral and Polarity(x.value + 1 if abs(x.value) < 3 else x.value),
    # не надежный
    PrefixCategory.INVERTION:
        lambda x: assert_not_neutral and Polarity(-sgn(x.value) * x.value),
    # очень надежный
    PrefixCategory.AMPLIFICATION:
        lambda x: assert_not_neutral and Polarity(x.value + sgn(x.value) if abs(x.value) < 3 else x.value),
    # не очень надежный
    (PrefixCategory.INVERTION, PrefixCategory.AMPLIFICATION):
        lambda x: assert_not_neutral and Polarity(-sgn(x.value))
}


class PosType(AutoEnum):
    A = ()
    ADV = ()
    ADVPRO = ()
    ANUM = ()
    APRO = ()
    COM = ()
    CONJ = ()
    INTJ = ()
    NUM = ()
    PART = ()
    PR = ()
    S = ()
    SPRO = ()
    V = ()
    UNKN = ()


class Token:
    id = None
    text_left_bound = None
    text_right_bound = None
    text = None
    type = None
    pos = None
    feats = None
    lemma = None
    root = False
    polarity = Polarity.NEUTRAL
    entity = False
    tonal_facts = None
    context_tokens = None
    group = None
    native_group = None
    inverted = False
    ampl_token = None

    def __init__(self, id: int, left: int, right: int, text: str, type: TokenType):
        self.id = id
        self.text_left_bound = left
        self.text_right_bound = right
        self.text = text
        self.type = type
        self.tonal_facts = []
        self.context_tokens = []
        self.group = []
        self.native_group = []

    def __str__(self):
        return '%d %s' % (self.id, self.text)

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.text_left_bound == other.text_left_bound and self.text_right_bound == other.text_right_bound \
                   and self.text == other.text
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.text_left_bound, self.text_right_bound, self.text))


class Sentence:
    id = None
    text_left_bound = None
    text_right_bound = None
    text = None
    tokens = None
    dep_tree = None
    word_ids = None
    facts = None

    def __init__(self, id: int, left: int, right: int, text: str):
        self.id = id
        self.text_left_bound = left
        self.text_right_bound = right
        self.text = text
        self.tokens = []
        self.word_ids = set()
        self.facts = []

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.text_left_bound == other.text_left_bound and self.text_right_bound == other.text_right_bound \
            and self.id == other.id

    def __str__(self):
        return str(self.id) + ' ' + self.text

    __repr__ = __str__


class Edge:
    type = None
    token = None
    orientation = None

    def __init__(self, type: SynType, token: Token, orientation: Orientation):
        self.type = type
        self.token = token
        self.orientation = orientation

    def __str__(self):
        return ' '.join([self.orientation.name, self.token.__str__(), self.type.name])

    __repr__ = __str__


def dd():
    return defaultdict(list)


class Node:
    text = None
    token_text = None
    polarity = Polarity.NEUTRAL
    weight = 0
    impact = 0
    both_connected = False
    entries = None
    process = True

    def __init__(self, token: Token, strict_ee: bool):
        self.text = token.lemma.lower()
        self.token_text = token.text.lower()
        if not strict_ee:
            self.text = self.text.replace('ё', 'е')
        self.entries = defaultdict(dd)

    def __str__(self):
        return self.text + ' ' + self.polarity.name

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.text == other.text
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.text)


class WeightedEdge:
    node = None
    weight = 0
    polarities = None

    def __init__(self, node: Node, type: bool=None, polarity_pair=None, weight: int=0):
        self.node = node
        self.polarities = defaultdict(list)
        if polarity_pair:
            self.polarities[type].append(polarity_pair)
        self.weight = weight

    def __eq__(self, other):
        if isinstance(other, WeightedEdge):
            return self.node == other.node
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.node)

    def __str__(self):
        return str(self.node) + ': ' + str(self.polarities)

    __repr__ = __str__


class State:
    stack = None
    closed = None
    paths = None
    states = None
    extended = None
    new_paths = None

    def __init__(self):
        self.stack = []
        self.closed = set()
        self.paths = []
        self.states = []
        self.extended = False
        self.new_paths = 0


class BaseStep:
    optional = False
    meta = None

    def __init__(self, optional=False, meta=None):
        self.optional = optional
        self.meta = meta


class Step(BaseStep):
    orientations = None
    edge_types = None
    token_check = None

    def __init__(self, orientations: Set[Orientation]=None, edge_types: Set[SynType]=None,
                 token_check=None, optional=False, meta=None):
        super().__init__(optional, meta)
        self.orientations = orientations
        self.edge_types = edge_types
        self.token_check = token_check

    def check(self, edge: Edge):
        return (self.orientations is None or edge.orientation in self.orientations) \
            and (self.edge_types is None or edge.type in self.edge_types) \
            and (self.token_check is None or self.token_check(edge.token))


class SimplePath(BaseStep):
    steps = None
    desc = None

    def __init__(self, steps: List[BaseStep], optional: bool=None, meta=None, desc=None):
        super().__init__(optional, meta)
        self.steps = steps
        self.meta = meta
        self.optional = optional if optional is not None else all(step.optional for step in steps)
        self.desc = desc


class BranchingStep(BaseStep):
    paths = None

    def __init__(self, paths: List[SimplePath], optional: bool=None, meta=None):
        super().__init__(optional if optional is not None else all(path.optional for path in paths), meta)
        self.paths = paths


class OR(BranchingStep):

    def __init__(self, paths: List[SimplePath], optional: bool=None, meta=None):
        super().__init__(paths, optional, meta)


class AND(BranchingStep):

    def __init__(self, paths: List[SimplePath], optional: bool=None, meta=None):
        super().__init__(paths, optional, meta)


class MatchingPath:
    path = None
    meta = None

    def __init__(self, path: List[Token]=None):
        self.path = path
        self.meta = {}

    def __eq__(self, other):
        return self.path == other.path

    def __str__(self):
        return str(self.path) + ': ' + str(self.meta.items())

    __repr__ = __str__


def time_wrap():
    def decorate(func):
        def call(*args, **kwargs):
            exec_time_millis = round(time.time() * 1000)
            result = func(*args, **kwargs)
            exec_time_millis = round(time.time() * 1000) - exec_time_millis
            print(
                '[ ' + func.__name__.upper() + ' ] COMPLETED IN %d MIN %d SEC %d MSEC' % (exec_time_millis / 1000 / 60,
                                                                                          exec_time_millis / 1000 % 60,
                                                                                          exec_time_millis % 1000))
            return result
        return call
    return decorate