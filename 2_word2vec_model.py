# Manejo de datos
import json
import re
# Manejo de datos en estructura de tablas
import pandas as pd
# Modelos
from gensim.models import word2vec

trainrecipes = json.load(open("./data/train.json",'r'))

def get_sentences(recipes):
  sentences = list()
  for recipe in recipes:
    clean_recipe = list()
    for ingredient in recipe['ingredients']:
      # Eliminamos la descripción de los ingredientes.
      ingredient =  re.sub(r'\(.*oz.\)|\%|0-9|crushed|crumbles|ground|minced|powder|fresh|powdered|chopped|sliced', '', ingredient)
      clean_recipe.append(ingredient.strip())
    sentences.append(clean_recipe)
  return sentences

sentences_train = get_sentences(trainrecipes)

# Set values for NN parameters
num_features = 300    # Word vector dimensionality 
min_word_count = 3    # 50% of the corpus                    
num_workers = 4       # Number of CPUs
context = 10          # Context window size;
downsampling = 1e-3   # threshold for configuring which higher-frequency words are randomly downsampled

# Initialize and train the model 
model = word2vec.Word2Vec(sentences_train, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

model_name = "./features_word2vec"
model.save(model_name)