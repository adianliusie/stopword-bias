from src.system_loader import SystemLoader, EnsembleLoader
from src.utils.evaluation import get_accuracy

system = EnsembleLoader('trained_models/imdb/baseline')
preds = system.load_formatted_preds(formatting='shuffle_stopwords', data_name='rt', mode='test')
labels = SystemLoader.load_labels('rt', mode='test')

print(get_accuracy(preds, labels))

