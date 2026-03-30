# MobileNetV2 Complete Example with Compile, Fit and Predict

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load pretrained MobileNetV2 model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save model
model.save('mobilenet_model.h5')

# ------------------------------
# Predict for a new image
# ------------------------------

# Class names according to folder order
class_names = ['class1', 'class2', 'class3', 'class4', 'class5']

# Load image
img = load_img('test_image.jpg', target_size=(224, 224))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

print('Predicted Class:', class_names[predicted_class])
print('Confidence:', confidence)


# ==========================================================
# ResNet50 Complete Example with Compile, Fit and Predict
# ==========================================================

from tensorflow.keras.applications import ResNet50

# Load pretrained ResNet50 model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build final model
resnet_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile model
resnet_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = resnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save model
resnet_model.save('resnet50_model.h5')

# Predict for a new image
prediction = resnet_model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

print('Predicted Class:', class_names[predicted_class])
print('Confidence:', confidence)
