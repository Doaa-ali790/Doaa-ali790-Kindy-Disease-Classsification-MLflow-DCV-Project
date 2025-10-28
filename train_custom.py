import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# === مسار البيانات ===
data_dir = r"C:\Users\PC\projects\Kidney-Disease-Classification-Deep-Learning-Project\artifacts\data_ingestion\data\kidney-ct-scan-image"

# === إعداد المولد للصور (تدريب + اختبار) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# === بناء النموذج ===
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # لأن لدينا فئتين: Normal و Tumor
])

# === إعداد النموذج للتدريب ===
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === تدريب النموذج ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# === حفظ النموذج المدرب ===
output_path = r"C:\Users\PC\projects\Kidney-Disease-Classification-Deep-Learning-Project\artifacts\training\model_trained_local.h5"
model.save(output_path)

print(f"\nتم حفظ النموذج بنجاح في: {output_path}")
