from src.system_loader import EnsembleLoader, SystemLoader
from src.utils.evaluation import get_accuracy

system = EnsembleLoader('trained_models/imdb/spurious')
preds  = system.load_preds('imdb',  mode='test')
labels = SystemLoader.load_labels('imdb', mode='test')
print(get_accuracy(preds, labels))


