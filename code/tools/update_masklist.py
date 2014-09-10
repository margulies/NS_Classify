import glob, os
masks = glob.glob("../masks/ns_kmeans_all/kmeans_all_60//*")
mask_names = [os.path.basename(os.path.splitext(os.path.splitext(mask)[0])[0]) for mask in masks]
masklist = zip(masks, mask_names)
masklist.sort(key=lambda x: int(x[1]))
c_60_a_t.masklist = masklist
c_60_a_t.save("../results/ns_60_RidgeClassifier_DM_abstract_topics_t_0.1/classifier.pkl")