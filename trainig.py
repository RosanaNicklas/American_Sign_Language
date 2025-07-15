import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from pathlib import Path

# === Parámetros ===
IMG_SIZE = 64
DATA_DIR = 'asl_alphabet_train'
CATEGORIES = sorted(os.listdir(DATA_DIR))

def load_data(img_size=64, limit_per_class=None):
    X, y = [], []
    for idx, category in enumerate(CATEGORIES):
        folder = Path(DATA_DIR) / category
        imgs = sorted(folder.iterdir())
        if limit_per_class:
            imgs = imgs[:limit_per_class]          # para debug rápido
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:                # <-- imagen corrupta o mal leída
                print(f"[WARN] Saltando {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size), cv2.INTER_AREA)
            X.append(img)
            y.append(idx)
    X = np.asarray(X, dtype="float32") / 255.0
    y = to_categorical(y, num_classes=len(CATEGORIES))
    return X, y


print("Cargando datos...")
X, y = load_data()
X = X / 255.0  # Normalización

# === Dividir datos ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Crear Modelo CNN ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Entrenamiento ===
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# === Guardar Modelo ===
model.save("asl_model.h5")
print("Modelo guardado como 'asl_model.h5'")


# === Evaluación ===
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title("Precisión del modelo")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.show()
