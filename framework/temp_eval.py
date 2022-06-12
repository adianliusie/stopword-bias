from src.system_loader import SystemLoader
from src.utils.evaluation import get_accuracy

system = SystemLoader('trained_models/sst/bert_rand')

preds  = system.load_preds('imdb',  mode='test')
labels = system.load_labels('imdb', mode='test')
print(get_accuracy(preds, labels))

preds  = system.load_preds('rt',  mode='test')
labels = system.load_labels('rt', mode='test')
print(get_accuracy(preds, labels))

preds  = system.load_preds('sst',  mode='test')
labels = system.load_labels('sst', mode='test')
print(get_accuracy(preds, labels))

system = SystemLoader('trained_models/sst/bert_baseline')

preds  = system.load_preds('imdb',  mode='test')
labels = system.load_labels('imdb', mode='test')
print(get_accuracy(preds, labels))

preds  = system.load_preds('rt',  mode='test')
labels = system.load_labels('rt', mode='test')
print(get_accuracy(preds, labels))

preds  = system.load_preds('sst',  mode='test')
labels = system.load_labels('sst', mode='test')
print(get_accuracy(preds, labels))

