import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import legacy
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# Settings
IMG_SIZE = (64, 64)     # Resize all images to 64x64
CHANNELS = 1            # Grayscale images
BATCH_SIZE = 8
EPOCHS = 20

# Dataset paths
TRAIN_DIR = r"C:\Users\Srimathi\ai-ehr-project\gen-ai_img_model\chest_xray\train"
VAL_DIR = r"C:\Users\Srimathi\ai-ehr-project\gen-ai_img_model\chest_xray\val"
TEST_DIR = r"C:\Users\Srimathi\ai-ehr-project\gen-ai_img_model\chest_xray\test"

# Model save path
MODEL_PATH = "medgan_generator_m2.h5"

# Dataset Loader
def load_dataset(folder):
    """Loads and preprocesses grayscale images from folder."""
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
                img = np.expand_dims(img, axis=-1)
                images.append(img)
            except Exception as e:
                print(f"Skipped {img_file}: {e}")
    return np.array(images)

# Load datasets
train_imgs = load_dataset(TRAIN_DIR)
val_imgs = load_dataset(VAL_DIR)
test_imgs = load_dataset(TEST_DIR)
print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

# Generator Network
def build_generator():
    """Builds CNN generator for image enhancement."""
    input_img = Input(shape=(*IMG_SIZE, CHANNELS))
    x = Conv2D(64, 3, padding='same')(input_img)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(CHANNELS, 3, padding='same', activation='sigmoid')(x)
    return Model(input_img, x, name="Generator")

# Discriminator Network
def build_discriminator():
    """Builds CNN discriminator to classify real vs generated images."""
    model = Sequential(name="Discriminator")
    model.add(Conv2D(64, 3, strides=2, padding='same', input_shape=(*IMG_SIZE, CHANNELS)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 3, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(1, 3, padding='same', activation='sigmoid'))
    return model

# GAN Training Function
def train(train_data):
    """Trains GAN model (Generator + Discriminator)."""
    generator = build_generator()
    discriminator = build_discriminator()
    optimizer = legacy.Adam(0.0002, 0.5)

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Combined model (Generator + frozen Discriminator)
    input_noisy = Input(shape=(*IMG_SIZE, CHANNELS))
    generated_img = generator(input_noisy)
    discriminator.trainable = False
    validity = discriminator(generated_img)
    combined = Model(input_noisy, [validity, generated_img])
    combined.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1e-3, 1], optimizer=optimizer)

    batch_count = int(len(train_data) / BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for i in range(batch_count):
            idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
            clean_batch = train_data[idx]
            noisy_batch = clean_batch + 0.1 * np.random.normal(size=clean_batch.shape)
            noisy_batch = np.clip(noisy_batch, 0., 1.)

            # Train discriminator
            disc_out_shape = discriminator.predict(clean_batch[:1]).shape[1:]
            valid_labels = np.ones((BATCH_SIZE, *disc_out_shape))
            fake_labels = np.zeros((BATCH_SIZE, *disc_out_shape))

            d_loss_real = discriminator.train_on_batch(clean_batch, valid_labels)
            d_loss_fake = discriminator.train_on_batch(generator.predict(noisy_batch), fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            g_loss = combined.train_on_batch(noisy_batch, [valid_labels, clean_batch])

            if i % 50 == 0 or i == batch_count - 1:
                print(f"[Batch {i}/{batch_count}] D loss: {d_loss[0]:.4f}, D acc: {d_loss[1]:.4f}, G loss: {g_loss[0]:.4f}")

    generator.save(MODEL_PATH)
    print(f"\nTraining complete. Model saved as: {MODEL_PATH}")
    return generator

# Load or Train Model
if os.path.exists(MODEL_PATH):
    print(f"Found existing model: {MODEL_PATH}. Skipping training...")
    generator = load_model(MODEL_PATH, compile=False)
else:
    print("No trained model found. Starting training process...")
    generator = train(train_imgs)

# Testing and Saving Results
OUTPUT_DIR = "enhanced_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_generator(generator, test_data, num_samples=5):
    """Evaluates and saves generator results to disk."""
    noisy_test = test_data + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
    noisy_test = np.clip(noisy_test, 0., 1.)
    generated_imgs = generator.predict(noisy_test)

    psnr_vals, ssim_vals = [], []
    for i in range(len(test_data)):
        psnr_vals.append(psnr(test_data[i], generated_imgs[i]))
        ssim_vals.append(ssim(test_data[i].squeeze(), generated_imgs[i].squeeze(), data_range=1.0))

    print(f"\nAverage PSNR: {np.mean(psnr_vals):.2f}, Average SSIM: {np.mean(ssim_vals):.4f}")
    print(f"Saving generated images to folder: {OUTPUT_DIR}\n")

    for i in range(num_samples):
        orig = (test_data[i].squeeze() * 255).astype(np.uint8)
        noisy = (noisy_test[i].squeeze() * 255).astype(np.uint8)
        enhanced = (generated_imgs[i].squeeze() * 255).astype(np.uint8)

        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "original.png"), orig)
        cv2.imwrite(os.path.join(sample_dir, "noisy.png"), noisy)
        cv2.imwrite(os.path.join(sample_dir, "enhanced.png"), enhanced)

        plt.figure(figsize=(8, 3))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(orig, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Noisy")
        plt.imshow(noisy, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Enhanced")
        plt.imshow(enhanced, cmap="gray")
        plt.axis("off")
        plt.show()

    print("Image results saved successfully!")

# Run testing
test_generator(generator, test_imgs)
