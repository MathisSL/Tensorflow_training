#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


(x_train, y_train),(x_test, y_test) = mnist.load_data()
r = np.random.randint(0, len(x_train))
image = x_train[r]
label = y_train[r]
plt.imshow(image)
plt.title(label)
plt.show


# In[8]:


x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255 
print(x_train.shape)


# In[10]:


class Dense(layers.Layer):
    def __init__(self, units) :
        super(Dense, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(
            name = 'w',
            shape = (input_shape[-1], self.units),
            initializer = 'random_normal',
            trainable = True,
        )
    
        self.b = self.add_weight(
            name = 'b',
            shape = (self.units, ),
            initializer = 'zeros',
            trainable = True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b   

class my_model(keras.Model):
    def __init__(self,num_classes = 10):
        super(my_model, self).__init__()
        self.dense1 = Dense(64)
        self.flatten = layers.Flatten()
        self.dense2 = Dense(num_classes) 
    def call(self, input_tensor):
        x = tf.nn.relu(self.dense1(input_tensor))
        x = self.flatten(x)
        return self.dense2(x)  


# In[14]:


# Initialize the model
model = my_model()

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)

# Evaluate the model
model.evaluate(x_test, y_test)

# Print the model summary
model.summary()


# In[46]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.legend(['Precision','Loss'])
plt.xlabel('epochs')
plt.ylabel('%')

plt.title('Tracé de la précision et de la fonctiond de perte')
plt.show()


# In[60]:


y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis = 1)
accuracy = sum(y_pred == y_test)/len(y_test)
print(f' La précision du model est de {accuracy*100} %')


# In[74]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_pred,y_test)
# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # cm is the confusion matrix, annot=True adds the numbers to the plot, fmt='d' formats the numbers as integers
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[98]:


misspred = np.where(y_pred != y_test)[0]
i = np.random.choice(misspred)
plt.imshow(x_test[i])
plt.title(f' Vrai label : {y_test[i]} prédiction {y_pred[i]}')
plt.show()

