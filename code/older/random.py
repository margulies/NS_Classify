# {'max_features': np.linspace(2, 40, 4).astype(int), 
# 		'n_estimators': np.round(np.linspace(5, 141, 4)).astype(int),'learning_rate': np.linspace(0.05, 1, 4).astype('float')}
#



	# for num, c in enumerate(clf.fit_clfs.flatten()):
	# 	flag = False
	# 	if c:
	# 		if not flag:
	# 			flag = True
	# 			res = c.best_params_
	# 			for item in res:
	# 				res[item] = [res[item]]
	# 		else:
	# 			for item in res:
	# 				res[item] = res[item].append(c.best_params_[item])

	# for item in res:
	# 	print item
	# 	print np.array(res[item]).mean()