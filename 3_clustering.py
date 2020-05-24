# Manejo de datos
import re
import collections
import json

# Manejo de datos en estructura de tablas
import numpy as np

# Modelo
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

# Visualización
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

trainrecipes = json.load(open("./data/train.json",'r'))

model_name = "./features_word2vec"
model = Word2Vec.load(model_name)

corpus = sorted(model.wv.vocab.keys())
emb_tuple = tuple([model.wv[v] for v in corpus])
X = np.vstack(emb_tuple)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(15,15))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

# Extraer ingredientes de las recetas (Eliminar cantidades, medidas, etc.)
raw_ingredients = list()

for recipe in trainrecipes:
  for ingredient in recipe['ingredients']:
    raw_ingredients.append(re.sub(r'\(.*oz.\)|\%|0-9|crushed|crumbles|ground|minced|powder|fresh|powdered|chopped|sliced', '', ingredient).strip())

# Extraemos la cocina de cada receta
raw_cuisines = list()
for recipe in trainrecipes:
  raw_cuisines.append(recipe['cuisine'].strip())

    
# Conteo de la frecuencia en los tipos de cocina
# return dictionary key:value (value:frecuency)
counts_ingr = collections.Counter(raw_ingredients) 
counts_cuis = collections.Counter(raw_cuisines)

# Etiquetar los ingredientes por frecuencia. 
track_ingredients = dict(zip(counts_cuis.keys(), [list() for x in counts_cuis.keys()]))
for recipe in trainrecipes:
  clean_recipe = list()
  for ingredient in recipe['ingredients']:
    # Eliminamos la descripción de los ingredientes.
    ingredient =  re.sub(r'\(.*oz.\)|\%|0-9|crushed|crumbles|ground|minced|powder|fresh|powdered|chopped|sliced', '', ingredient)
    clean_recipe.append(ingredient.strip())
      
  track_ingredients[recipe['cuisine']].extend(clean_recipe)

for label, tracking in track_ingredients.items():
  track_ingredients[label] = collections.Counter(tracking)

def return_most_popular(v):
  cuisine = None
  record = 0
  for label, tracking in track_ingredients.items():
    norm_freq = float(tracking[v]) / float(counts_cuis[label])
    if norm_freq > record:
      cuisine = label
      record = norm_freq
  return cuisine

track_2color = {
  u'irish':"#000000", # Negro
  u'mexican':"#FFFF00", # Amarillo
  u'chinese':"#1CE6FF", # Cyan
  u'filipino': "#FF34FF", # Rosa 
  u'vietnamese':"#FF4A46", # Rojo
  u'spanish':"#FFC300",  # Verde Bosque
  u'japanese':"#006FA6", # Azul Oseano
  u'moroccan':"#A30059",# Morado
  u'french':"#FFDBE5",  # Rosa claro
  u'greek': "#7A4900",  # Café 
  u'indian':"#0000A6", # Azul Marino 
  u'jamaican':"#63FFAC", # Verde Acua
  u'british': "#B79762", # Café claro
  u'brazilian': "#EEC3FF", # Morado Claro
  u'russian':"#8FB0FF", # Azul Claro 
  u'cajun_creole':"#997D87", # Violeta
  u'thai':"#5A0007", # Café Rojo
  u'southern_us':"#809693", # Gris
  u'korean':"#FEFFE6", # Amarillo Claro
  u'italian':"#1B4400" # Verde Militar
}

color_vector = list()
for v in corpus:
  cuisine = return_most_popular(v)
  color_vector.append(track_2color[cuisine])

lgend = list()
for l, c in track_2color.items():
  lgend.append(mpatches.Patch(color=c, label=l))

sns.set_context("poster")
fig, ax = plt.subplots(figsize=(15,15))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_vector, alpha=.6, s=60)
plt.legend(handles=lgend, loc='lower center', ncol=5, fontsize='xx-small', frameon=False)
plt.show()
