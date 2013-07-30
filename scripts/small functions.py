#Flatten cv_scores and turn into array
cv_scores = []
for d1 in yeoClass.fit_clfs:
    for d2 in d1:
    	if d2:
	        cv_scores.append(d2.cv_scores_)
cv_scores = np.array(cv_scores)

avg_cv_scores = cv_scores.mean(axis =0)

# Plot accuracy vs number of features across all
import pylab as pl
pl.figure()
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (nb of misclassifications)")
pl.plot(xrange(1, len(avg_cv_scores) + 1), avg_cv_scores)
pl.show()

# correlate two types of importances
cors = []
for a in range(0, 6):
	for b in range(0, 6):
		if not a == b:
			cors.append(tuple(pearsonr(linSVM_rank[a, b], linSVM_imp[a, b])))

np.abs(zip(*cors)[0]).mean()

# Make linSVMcoefs absolute (more like feature importances)
for a in range(0, 6):
	for b in range(0, 6):
		if not a == b:
			linSVM_abs[a, b] = np.abs(np.array(linSVM_imp[a, b]))

# Plots for each region and all regions of important features
for i in range(1, 7):
    yeoClass.plot_importances(i-1, file_name="../results/Yeo_SVMcoef_"+str(i)+".png", thresh=10)
yeoClass.plot_importances(None, file_name="../results/Yeo_SVMcoef_overall.png", thresh=05)

        imps = [(i, yeoClass.feature_names[num]) for (num, i) in
                enumerate(fi)]

