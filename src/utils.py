import os
import numpy as np
def loadGloVe(fPath):
    path_to_glove_file = os.path.join(fPath)

    preTGloVe_wv = {}
    with open(path_to_glove_file, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            preTGloVe_wv[word] = coefs

    print("Found %s word vectors." % len(preTGloVe_wv))
    return preTGloVe_wv