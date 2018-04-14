import itertools

from scripts.polarity import get_p_polarity, get_s_polarity
from scripts.typez_and_constants import Polarity

combinations = list(itertools.product([Polarity.POSITIVE, Polarity.NEGATIVE, None], repeat=2))
polarities = [Polarity.POSITIVE, Polarity.NEGATIVE]
exceptions = []
rules = []
for dep_noun1, adj1 in combinations:
    for dep_noun2, adj2 in combinations:
        invert = False
        for noun1 in polarities:
            polarity1 = get_p_polarity(adj1, get_s_polarity(dep_noun1, noun1))
            found = None
            for noun2 in polarities:
                polarity2 = get_p_polarity(adj2, get_s_polarity(dep_noun2, noun2))
                if polarity1 == polarity2:
                    if found is not None:
                        exceptions.append((dep_noun1, adj1, noun1, dep_noun2, adj2, noun2))
                        found = None
                        break
                    else:
                        found = noun2
            if found is None:
                break
            else:
               invert ^= (noun1 != found)
        if invert:
            exceptions.append((dep_noun1, adj1, dep_noun2, adj2))
        else:
            rules.append((dep_noun1, adj1, dep_noun2, adj2, noun1 == noun2))

for e in exceptions:
    print(e)