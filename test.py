from textattack.datasets.classification import AGNews
from textattack.models.classification.lstm import LSTMForAGNewsClassification
model = LSTMForAGNewsClassification()
from textattack.goal_functions import UntargetedClassification
goal_function = UntargetedClassification(model)

from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.shared import Attack

from textattack.transformations import WordSwap

class BananaWordSwap(WordSwap):
    def _get_replacement_words(self, word):
        return ['banana']

import nltk
import functools

@functools.lru_cache(maxsize=2**14)
def get_entities(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged, binary=True)
    return entities.leaves()

from textattack.constraints import Constraint

class NamedEntityConstraint(Constraint):
    def _check_constraint(self, transformed_text, current_text, original_text=None):
        transformed_entities = get_entities(transformed_text.text)
        current_entities = get_entities(current_text.text)
        if len(current_entities) == 0:
            return False
        if len(current_entities) != len(transformed_entities):
            return False
        else:
            current_word_label = None
            transformed_word_label = None
            for (word_1, label_1), (word_2, label_2) in zip(current_entities, transformed_entities):
                if word_1 != word_2:
                    if (label_1 not in ['NNP', 'NE']) or (label_2 not in ['NNP', 'NE']):
                        return False
            return True

transformation = BananaWordSwap()
constraints = [RepeatModification(), StopwordModification(), NamedEntityConstraint()]
search_method = GreedySearch()
attack = Attack(goal_function, constraints, transformation, search_method)

print(attack)

import torch
torch.cuda.is_available()
from tqdm import tqdm
from textattack.loggers import CSVLogger
from collections import deque
worklist = deque(range(0, 10))
results_iterable = attack.attack_dataset(AGNews(), indices=worklist)
# results_iterable = attack.attack_dataset(AGNews())
results = []

logger = CSVLogger(color_method='html')

for result in tqdm(results_iterable, total=10):
    logger.log_attack_result(result)
    results.append(result)

# import pandas as pd
# pd.options.display.max_colwidth = 480

for j, result in enumerate(results):
    print(j)
    print("ori: ", result.original_result.attacked_text)
    print("adv: ", result.perturbed_result.attacked_text)

# from IPython.core.display import display, HTML
# display(HTML(logger.df[['original_text', 'perturbed_text']].to_html(escape=False)))
# logger.df[['original_text', 'perturbed_text']]



sentence = 'Jack Black starred in the 2003 film classic "School of Rock".'
print(get_entities(sentence))
sentence = ('In 2017, star quarterback Tom Brady led the Patriots to the Super Bowl, but lost to the Philadelphia Eagles.')


named_entities = [entity for entity in entities if isinstance(entity, nltk.tree.Tree)]
print(named_entities)
