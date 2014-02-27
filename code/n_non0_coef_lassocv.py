from sklearn.linear_model import LassoCV
import cPickle
from regionalClassifier import *


def LassoCV_parallel(args):
	c_data, pair = args

	index = (pair[0][1], pair[1][1]) # Tuple numeric index of pairs
	names = [pair[0][0], pair[1][0]] # Actual paths to masks
	
	X, y = c_data[index]

	L = LassoCV().fit(X, y)

	return np.where(L.coef_ != 0)[0].shape[0], L.alpha_


clf = cPickle.load(open("../results/ns_k_20_Ridge_Xt_0.1_topics_t_0.07/classifier.pkl", 'rb'))

mask_pairs = list(itertools.permutations(clf.masklist, 2))

update_progress(0)

p = Pool(processes = 8)

prog = 0.0
total = len(list(mask_pairs))

alphas = []
ns = []

for n, alpha in p.imap_unordered(LassoCV_parallel, itertools.izip(itertools.repeat(clf.c_data), mask_pairs)):
	ns.append(n)
	alphas.append(alpha)


	prog = prog + 1
	update_progress(int(prog / total * 100))


