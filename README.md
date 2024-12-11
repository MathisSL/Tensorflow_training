# Tensorflow_training
# Différence entre Subclassing API et Sequential API dans Keras

Dans Keras, il existe deux principales approches pour définir des modèles de réseaux de neurones : **Sequential** et **Subclassing**. Ces deux approches permettent de créer des modèles de manière différente, et chacune présente des avantages selon les besoins du projet.

## 1. **Sequential API**

L'API **Sequential** est la méthode la plus simple et linéaire pour construire un modèle. Dans cette approche, les couches sont empilées les unes après les autres, chaque couche prenant en entrée la sortie de la couche précédente et la transmettant à la suivante. Cette approche est idéale pour des modèles simples où chaque couche est connectée directement à la suivante.

### Exemple avec Sequential :

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(64, input_dim=8),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(1)
])
```
### Avantages de Sequential :

Facile à utiliser et à comprendre.
Idéal pour des modèles avec une architecture simple et linéaire.

## 2. Subclassing API

L'approche Subclassing consiste à créer une nouvelle classe qui hérite de tf.keras.Model et à définir explicitement la logique du modèle. On doit redéfinir la méthode call pour spécifier comment les données passent à travers les différentes couches du modèle. Cette méthode offre plus de flexibilité et de contrôle, ce qui est particulièrement utile pour des architectures complexes, comme celles avec des branches multiples ou des modèles personnalisés (par exemple, les GANs).
### Exemple avec Subclassing :

```python
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

model = MyModel()
```
### Avantages de Subclassing :

Offre une flexibilité totale pour définir des architectures complexes et des comportements personnalisés.
Permet de définir des modèles avec des logiques non linéaires (par exemple, plusieurs branches, ou des couches récurrentes).

## Comparaison

| Critère                         | **Sequential API**                     | **Subclassing API**                   |
|---------------------------------|----------------------------------------|---------------------------------------|
| **Simplicité**                  | Facile à utiliser, approche linéaire   | Plus complexe, mais plus flexible     |
| **Contrôle sur l'architecture** | Limité à un seul flux de données       | Entièrement personnalisable           |
| **Idéal pour**                  | Modèles simples et linéaires           | Modèles complexes et personnalisés    |





