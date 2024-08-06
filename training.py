import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
import random

# Function to check if a file is a valid image
def is_image_file(filename):
    try:
        with Image.open(filename) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

# Paths
original_dataset_dir = 'kagglecatsanddogs_5340/PetImages'
base_dir = 'dataset'

# Create directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.makedirs(train_cats_dir, exist_ok=True)
os.makedirs(train_dogs_dir, exist_ok=True)
os.makedirs(validation_cats_dir, exist_ok=True)
os.makedirs(validation_dogs_dir, exist_ok=True)

# List of filenames
cat_dir = os.path.join(original_dataset_dir, 'Cat')
dog_dir = os.path.join(original_dataset_dir, 'Dog')

cat_filenames = [f for f in os.listdir(cat_dir) if f.endswith('.jpg') and is_image_file(os.path.join(cat_dir, f))]
dog_filenames = [f for f in os.listdir(dog_dir) if f.endswith('.jpg') and is_image_file(os.path.join(dog_dir, f))]

# Shuffle the data
random.shuffle(cat_filenames)
random.shuffle(dog_filenames)

# Define split sizes
train_size = int(0.8 * len(cat_filenames))  # 80% for training
validation_size = len(cat_filenames) - train_size  # 20% for validation

# Copy files to train and validation directories
for i in range(train_size):
    shutil.copyfile(os.path.join(cat_dir, cat_filenames[i]), os.path.join(train_cats_dir, cat_filenames[i]))
    shutil.copyfile(os.path.join(dog_dir, dog_filenames[i]), os.path.join(train_dogs_dir, dog_filenames[i]))

for i in range(train_size, len(cat_filenames)):
    shutil.copyfile(os.path.join(cat_dir, cat_filenames[i]), os.path.join(validation_cats_dir, cat_filenames[i]))
    shutil.copyfile(os.path.join(dog_dir, dog_filenames[i]), os.path.join(validation_dogs_dir, dog_filenames[i]))

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# # Only rescaling for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# # Calculate steps_per_epoch and validation_steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Print the calculated steps to ensure they are reasonable
print(f"train_generator.samples : {train_generator.samples}")
print(f"train_generator.batch_size : {train_generator.batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# # Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# # Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save the model immediately after training
model.save('cats_vs_dogs.h5')
print("Model saved as cats_vs_dogs.h5")

# Evaluate the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
