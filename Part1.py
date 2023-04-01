import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Spécifier les répertoires des données d'entraînement, de validation et de test
train_dir = 'path/to/train'
val_dir = 'path/to/val'
test_dir = 'path/to/test'

# Spécifier les dimensions des images et le nombre de classes
img_width, img_height = 256, 256
num_classes = 3

# Utiliser l'architecture de réseau VGG16 pré-entraînée sur ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Geler les poids des couches du modèle de base
for layer in base_model.layers:
    layer.trainable = False

# Ajouter des couches personnalisées pour notre problème de classification
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compiler le modèle avec l'optimiseur Adam et la perte de catégorisation croisée
model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Utiliser l'augmentation de données pour générer des images supplémentaires pour l'ensemble de formation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Utiliser l'augmentation de données pour l'ensemble de validation et de test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Préparer les générateurs de données pour l'ensemble de formation, de validation et de test
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=32,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(img_width, img_height),
                                                batch_size=32,
                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=32,
                                                   class_mode='categorical')

# Entraîner le modèle en utilisant les générateurs de données
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=10,
          validation_data=val_generator,
          validation_steps=len(val_generator))

# Évaluer le modèle sur l'ensemble de test et afficher les scores de précision, de rappel et de score F1 pour chaque classe
scores = model.evaluate(test_generator, steps=len(test_generator))
print("
