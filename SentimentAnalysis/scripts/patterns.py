from scripts.typez_and_constants import SimplePath, OR, Orientation, SynType, Step, Token, AND, Polarity, PosType
from scripts.utils import is_adj_or_part, is_conjunctive, is_adversative, is_comparative


def adj_not_comp_check(token: Token):
    return is_adj_or_part(token) and 'comp' not in token.feats


def noun_check(token: Token):
    return token.pos == PosType.S and not entity_check(token)


def verb_check(token: Token):
    return token.pos == PosType.V and 'partcp' not in token.feats and not entity_check(token)


def adj_comp_check(token: Token):
    return is_adj_or_part(token) and token.polarity != Polarity.NEUTRAL and 'comp' in token.feats


def entity_check(token: Token):
    return token.entity


def conj_comp_check(token: Token):
    return token.pos == PosType.CONJ and (is_conjunctive(token) or is_adversative(token) or is_comparative(token))


def part_check(token: Token):
    return token.pos == PosType.PR and token.text.lower() == 'у' or token.pos == PosType.S


def adv_check(token: Token):
    return token.pos in {PosType.ADV, PosType.APRO}


def adv_deps_helper(postfix: str=''):
    return SimplePath(steps=[
        Step(
            orientations={Orientation.OUT},
            edge_types={SynType.огранич, SynType.колич_огран, SynType.опред, SynType.обст},
            token_check=adv_check,
            meta='i' + postfix
        ),
        # (B)
        Step(
            orientations={Orientation.OUT},
            edge_types={SynType.огранич, SynType.колич_огран, SynType.опред},
            token_check=adv_check,
            meta='ii' + postfix,
            optional=True
        )
    ], optional=True)


def adj_deps_helper(optional: bool=False, postfix: str=''):
    # A(B)
    return SimplePath(steps=[
        # A
        Step(
            orientations={Orientation.OUT},
            edge_types={SynType.опред, SynType.оп_опред},
            token_check=adj_not_comp_check,
            meta='p' + postfix
        ),
       adv_deps_helper('p' + postfix)
    ], optional=optional)


# (S[A(B)]) & (A(B))
noun_deps = AND(paths=[
    # (S(A))
    SimplePath(steps=[
        # S
        Step(
            orientations={Orientation.OUT},
            edge_types={SynType._1_компл, SynType._2_компл, SynType._3_компл, SynType._4_компл, SynType._5_компл,
                        SynType.аппоз, SynType.об_аппоз, SynType.ном_аппоз},
            token_check=noun_check,
            meta='s'
        ),
        # (A)
        adj_deps_helper(optional=True, postfix='s')
    ], optional=True, desc="1"),
    # A
    adj_deps_helper(optional=True, postfix='o')
])

simple_patterns = [
    SimplePath(steps=[
        OR(paths=[
            # A(B)
            SimplePath(steps=[
                # A
                Step(
                    orientations={Orientation.IN},
                    edge_types={SynType.предик},
                    token_check=adj_not_comp_check,
                    meta='p'
                ),
                # (B)
                adv_deps_helper('p')
            ]),
            # A(B)
            adj_deps_helper(),
            # (V)(N[(S[A(B)]) & (A(B))]) | V(A(B))
            SimplePath(steps=[
                OR(paths=[
                    # N((S[A(B)]) & (A(B)))
                    SimplePath(steps=[
                        # N
                        OR(paths=[
                            SimplePath(steps=[
                                Step(
                                    orientations={Orientation.IN},
                                    edge_types={SynType.предик},
                                    token_check=noun_check,
                                    meta='o'
                                )
                            ]),
                            SimplePath(steps=[
                                Step(
                                    orientations={Orientation.OUT},
                                    edge_types={SynType.аппоз, SynType.об_аппоз, SynType.ном_аппоз},
                                    token_check=noun_check,
                                    meta='o'
                                )
                            ])
                        ]),
                        # (S[A(B)]) & (A(B))
                        noun_deps
                    ], desc="2"),

                    # V [((B) & N[(S[A(B)]) & (A(B))]) | (A(B))]
                    SimplePath(steps=[
                        # V
                        Step(
                            orientations={Orientation.IN},
                            edge_types={SynType.предик, SynType._1_компл},
                            token_check=verb_check,
                            meta='a'
                        ),
                        OR(paths=[
                            SimplePath(steps=[
                                AND(paths=[
                                    # (B)
                                    adv_deps_helper('a'),
                                    # (N[(S[A(B)]) & (A(B))])
                                    SimplePath(steps=[
                                        # N
                                        Step(
                                            orientations={Orientation.OUT},
                                            edge_types={SynType._1_компл, SynType.обст_тавт, SynType.об_обст,
                                                        SynType.суб_обст, SynType.обст, SynType.предик},
                                            token_check=noun_check,
                                            meta='o'
                                        ),
                                        # (S[A(B)]) & (A(B))
                                        noun_deps
                                    ], optional=True)
                                ], optional=False)
                            ]),
                            # (A(B))
                            SimplePath(steps=[
                                AND(paths=[
                                    # (B)
                                    adv_deps_helper('a'),
                                    SimplePath(steps=[
                                        Step(
                                            orientations={Orientation.OUT},
                                            edge_types={SynType.присвяз},
                                            token_check=adj_not_comp_check,
                                            meta='p',
                                        ),
                                        adv_deps_helper('p')
                                    ])
                                ])
                            ])
                        ], optional=True)
                    ], desc='4')
                ])
            ])
        ])
    ], meta='e'),
]

# [NE] OR [TW1 ([NE] OR [PR NE])]
cmp_branch = OR(paths=[
    SimplePath(steps=[
        Step(
            orientations={Orientation.OUT},
            edge_types={SynType.сравнит},
            token_check=entity_check,
            meta='ce'

        )
    ]),
    SimplePath(steps=[
        Step(
            orientations={Orientation.OUT},
            edge_types={SynType.сравнит},
            token_check=conj_comp_check
        ),
        OR(paths=[
            SimplePath(steps=[
                Step(
                    orientations={Orientation.OUT},
                    edge_types={SynType.сравн_союзн},
                    token_check=entity_check,
                    meta='ce'
                )
            ]),
            SimplePath(steps=[
                Step(
                    orientations={Orientation.OUT},
                    edge_types={SynType.сравн_союзн},
                    token_check=part_check
                ),
                Step(
                    orientations={Orientation.OUT},
                    edge_types={SynType.квазиагент, SynType.предл, SynType.атриб},
                    token_check=entity_check,
                    meta='ce'
                )
            ])

        ])
    ])
])

cmp_patterns = [
    # CMP
    SimplePath(steps=[
        Step(
            orientations={Orientation.IN},
            edge_types={SynType.квазиагент, SynType.предл, SynType.атриб},
            token_check=part_check,
            optional=True
        ),
        OR(paths=[
            SimplePath(steps=[
                OR(paths=[
                    SimplePath(steps=[
                        Step(
                            orientations={Orientation.IN},
                            edge_types={SynType.предик, SynType.обст},
                            token_check=noun_check,
                            meta='o'
                        )
                    ]),
                    SimplePath(steps=[
                        Step(
                            orientations={Orientation.IN},
                            edge_types={SynType.предик, SynType.обст},
                            token_check=verb_check,
                            meta='a'
                        )
                    ]),
                ]),
                AND(paths=[
                    SimplePath(steps=[
                        Step(
                            orientations={Orientation.OUT},
                            edge_types={SynType.предик},
                            token_check=noun_check,
                            meta='o',
                            optional=True
                        )
                    ]),
                    SimplePath(steps=[
                        Step(
                            orientations={Orientation.OUT},
                            edge_types={SynType.обст},
                            token_check=adj_comp_check,
                            meta='p'
                        ),
                        AND(paths=[
                            adv_deps_helper('p'),
                            SimplePath(steps=[
                                cmp_branch
                            ])
                        ])
                    ])
                ])
            ]),
            SimplePath(steps=[
                Step(
                    orientations={Orientation.IN},
                    edge_types={SynType.предик, SynType.обст},
                    token_check=adj_comp_check,
                    meta='p'
                ),
                AND(paths=[
                    adv_deps_helper('p'),
                    SimplePath(steps=[
                        Step(
                            orientations={Orientation.OUT},
                            edge_types={SynType.предик},
                            token_check=noun_check,
                            meta='o',
                            optional=True
                        )
                    ]),
                    SimplePath(steps=[
                        cmp_branch
                    ])
                ])
            ])
        ]),
    ], meta='e')
]