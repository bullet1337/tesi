from analyze import DICTS_PATH, defaultdict, CATEGORIES, sgn, Polarity

for pos in ['adv', 'adj', 'noun', 'verb']:
    words = defaultdict(set)
    for category in CATEGORIES:
        if category == 'test':
            continue

        with open(DICTS_PATH.format(category, pos), mode='r', encoding='utf8') as file:
            for line in file:
                w, p = line.strip().split()
                words[w].add((int(p), category))

    for w, p in words.items():
        if len(p) > 1 and len({sgn(e[0]) for e in p}) > 1:
            print(w, ['({} {})'.format(Polarity(e[0]).name, e[1]) for e in p])
        if any(e[0] > 0 for e in p) and any(e[0] < 0 for e in p):
            print('WOW')