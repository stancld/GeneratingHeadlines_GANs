import sys

terms = {}

THRESHOLD_TF = 1


def update(tokens):
    for token in tokens:
        if token in terms:
            terms[token] += 1
        else:
            terms[token] = 1


# go over terms and update their tf
for line in sys.stdin:
    update(line.split())

# save and pass terms
f = open(str(THRESHOLD_TF) + 'tf_terms.txt', 'w')

for term in list(terms):
    if terms[term] >= THRESHOLD_TF:
        print(term)
        f.write(term + '\n')

f.close()
