import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import legacy
from tqdm import tqdm

# -------------------------
# Settings
# -------------------------
IMG_SIZE = (64, 64)
CHANNELS = 1
BATCH_SIZE = 8
EPOCHS = 20

TRAIN_DIR = r"C:\Users\Srimathi\ai-ehr-project\gen-ai_img_model\chest_xray\train"
VAL_DIR = r"C:\Users\Srimathi\ai-ehr-project\gen-ai_img_model\chest_xray\val"
TEST_DIR = r"C:\Users\Srimathi\ai-ehr-project\gen-ai_img_model\chest_xray\test"

# -------------------------
# Dataset loader
# -------------------------
def load_dataset(folder):
    images = []
    print(f"Loading images from {folder}: ")
    for subfolder in os.listdir(folder):
        sub_path = os.path.join(folder, subfolder)
        if not os.path.isdir(sub_path):
            continue
        for img_file in os.listdir(sub_path):
            path = os.path.join(sub_path, img_file)
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=-1)  # add channel
                images.append(img)
            except Exception as e:
                print(f"Skipped {img_file}: {e}")
    return np.array(images)

train_imgs = load_dataset(TRAIN_DIR)
val_imgs = load_dataset(VAL_DIR)
test_imgs = load_dataset(TEST_DIR)

print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# -------------------------
# Generator
# -------------------------
def build_generator():
    input_img = Input(shape=(*IMG_SIZE, CHANNELS))

    x = Conv2D(64, 3, padding='same')(input_img)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(CHANNELS, 3, padding='same', activation='sigmoid')(x)

    return Model(input_img, x, name="Generator")

# -------------------------
# Discriminator
# -------------------------
def build_discriminator():
    model = Sequential(name="Discriminator")
    model.add(Conv2D(64, 3, strides=2, padding='same', input_shape=(*IMG_SIZE, CHANNELS)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 3, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(1, 3, padding='same', activation='sigmoid'))
    return model

# -------------------------
# Build models
# -------------------------
generator = build_generator()
discriminator = build_discriminator()

optimizer = legacy.Adam(0.0002, 0.5)

discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Combined model
input_noisy = Input(shape=(*IMG_SIZE, CHANNELS))
generated_img = generator(input_noisy)
discriminator.trainable = False
validity = discriminator(generated_img)

combined = Model(input_noisy, [validity, generated_img])
combined.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1e-3, 1], optimizer=optimizer)

# -------------------------
# Training loop
# -------------------------
def train(train_data):
    batch_count = int(len(train_data) / BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for i in range(batch_count):
            idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
            clean_batch = train_data[idx]
            noisy_batch = clean_batch + 0.1 * np.random.normal(size=clean_batch.shape)
            noisy_batch = np.clip(noisy_batch, 0., 1.)

            # Train discriminator
            disc_out_shape = discriminator.predict(clean_batch[:1]).shape[1:]  # dynamic shape
            valid_labels = np.ones((BATCH_SIZE, *disc_out_shape))
            fake_labels = np.zeros((BATCH_SIZE, *disc_out_shape))

            d_loss_real = discriminator.train_on_batch(clean_batch, valid_labels)
            d_loss_fake = discriminator.train_on_batch(generator.predict(noisy_batch), fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = combined.train_on_batch(noisy_batch, [valid_labels, clean_batch])

            if i % 50 == 0 or i == batch_count-1:
                print(f"[Batch {i}/{batch_count}] D loss: {d_loss[0]:.4f}, D acc: {d_loss[1]:.4f}, G loss: {g_loss[0]:.4f}")

    return generator

# -------------------------
# Start training
# -------------------------
generator = train(train_imgs)

# -------------------------
# Save model
# -------------------------
generator.save("medgan_generator_m2.h5")
print("Training finished. Generator saved as medgan_generator_m2.h5")
