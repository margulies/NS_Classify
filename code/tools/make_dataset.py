import neurosynth.base.dataset
from neurosynth.base.dataset import Dataset
print neurosynth.base.dataset.__file__
dataset = Dataset('../data/unprocessed/abstract/full_database_revised.txt')
dataset.add_features('../data/unprocessed/abstract/abstract_features.txt')
dataset.save('../data/datasets/dataset_abs_words_pandas.pkl')

dataset = Dataset('../data/unprocessed/abstract/full_database_revised.txt')
dataset.add_features('../data/unprocessed/abstract_topics/doc_features100.txt')
dataset.save('../data/datasets/dataset_abs_topics_pandas.pkl')