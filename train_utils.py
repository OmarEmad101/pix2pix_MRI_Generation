import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import pandas as pd
from tensorflow.summary import create_file_writer


def load_dataset(images_path, masks_path):
    images, masks = [], []
    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))
    for img_file, mask_file in zip(image_files, mask_files):
        img = np.load(os.path.join(images_path, img_file))
        mask = np.load(os.path.join(masks_path, mask_file))
        images.append(img)
        masks.append(mask)
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images = tf.expand_dims(images,axis=-1)
    masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    masks = tf.expand_dims(masks, axis=-1)
    return images, masks

# def resize(input_image, real_image, height, width):
#     with tf.device('/CPU:0'):  # Force resizing on CPU
#         input_image = tf.image.resize(input_image, [height, width],
#                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#         real_image = tf.image.resize(real_image, [height, width],
#                                      method=tf.image.ResizeMethod.BICUBIC)
#     return input_image, real_image


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result
    
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

# Pix2Pix Model Definitions
# Updated Generator for 128x128 images
def Generator():
    inputs = tf.keras.layers.Input(shape=[128, 128, 1])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 64)
        downsample(128, 4),  # (batch_size, 32, 32, 128)
        downsample(256, 4),  # (batch_size, 16, 16, 256)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(256, 4),  # (batch_size, 16, 16, 512)
        upsample(128, 4),  # (batch_size, 32, 32, 256)
        upsample(64, 4),  # (batch_size, 64, 64, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 128, 128, 1)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def generate_images(model, test_input, tar, save_path, step, apply_noise, prefix="step"):
    """
    Generates and saves images showing the model's progress.
    
    Args:
        model: The trained generator model.
        test_input: Input mask.
        tar: Ground truth MRI image.
        save_path: Folder where images will be saved.
        step: Current training step.
    """
    test_input = tf.squeeze(test_input, axis=0)  # Remove batch dim
    tar = tf.squeeze(tar, axis=0)  # Remove batch dim
    prediction = model(tf.expand_dims(test_input, axis=0), training=True)
    prediction = tf.squeeze(prediction, axis=0)  # Remove batch dim after prediction

    cmap_mask = ListedColormap(['black', 'white', 'gray', 'red', 'yellow'])

    plt.figure(figsize=(15, 5))
    display_list = [test_input, tar, prediction]
    titles = ['Input Mask', 'Ground Truth (Grayscale)', 'Predicted Image (Grayscale)']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])

        if i == 0:
            if apply_noise:
                plt.imshow(display_list[i], cmap='gray')
            else: 
                plt.imshow(display_list[i], cmap=cmap_mask, vmin=0, vmax=4)
        else:
            plt.imshow(display_list[i], cmap='gray')

        plt.axis('off')

    os.makedirs(save_path, exist_ok=True)  # Create the folder if it doesn't exist
    save_filename = os.path.join(save_path, f"{prefix}_{step}.png")
    plt.savefig(save_filename)
    plt.close()  # Close the plot to free memory
    print(f"Saved progress image: {save_filename}")

def apply_gaussian_noise_to_background(masks, noise_std, noise_max):
    """
    Applies Gaussian noise to background pixels (where mask == 0) without exceeding ±noise_max.

    Args:
    - masks (tf.Tensor): Tensor of shape (N, H, W, 1) containing the segmentation masks.
    - noise_std (float): Standard deviation of the Gaussian noise.
    - noise_max (float): Maximum absolute value of noise.

    Returns:
    - noisy_masks (tf.Tensor): New tensor with noise applied to the background.
    - noise_tensor (tf.Tensor): The actual noise applied.
    """
    # Generate Gaussian noise
    noise = tf.random.normal(shape=masks.shape, mean=0.0, stddev=noise_std)

    # Clip noise within [-noise_max, noise_max]
    noise = tf.clip_by_value(noise, -noise_max, noise_max)

    # Create mask for background pixels (where mask == 0)
    background_mask = tf.cast(masks == 0, tf.float32)

    # Apply noise only to background pixels
    noisy_masks = masks + (background_mask * noise)

    return noisy_masks

def train_pix2pix(steps, normalization, data_dir, save_dir, generator_loss_fn, discriminator_fn, use_conditional_disc, apply_noise=False, noise_std=0.2, noise_max=0.5):
    """
    Train the Pix2Pix model with specified parameters.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
    save_dir = os.path.join(base_dir, save_dir)  # Ensure save d is relative

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ✅ TensorBoard Summary Writer
    summary_writer = create_file_writer(log_dir)

    # CSV file to log losses
    loss_file = os.path.join(save_dir, "training_loss.csv")
    with open(loss_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Generator Loss", "Discriminator Loss", "PSNR", "SSIM"])  # CSV header
    
    train_images, train_masks = load_dataset(os.path.join(data_dir, "model_images", "train"),
                                             os.path.join(data_dir, "model_masks", "train"))
    test_images, test_masks = load_dataset(os.path.join(data_dir, "model_images", "test"),
                                           os.path.join(data_dir, "model_masks", "test"))
    
    # Apply Gaussian noise if enabled
    if apply_noise:
        print(f"Applying Gaussian noise with std={noise_std} and max={noise_max}...")
        train_masks = apply_gaussian_noise_to_background(train_masks, noise_std, noise_max)
        test_masks = apply_gaussian_noise_to_background(test_masks, noise_std, noise_max)

        
    # train_masks, train_images = resize(train_masks,train_images,256,256)
    # test_masks, test_images = resize(test_masks, test_images,256,256)
    
    print(f"Loaded {train_images.shape} training images and {test_images.shape} test images")
    print(f"Loaded {train_masks.shape} training masks and {test_masks.shape} test masks")


    # if normalization:
    #     def normalize(image):
    #         return 2 * (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image)) - 1
    #     train_images = normalize(train_images)
    #     test_images = normalize(test_images)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_masks, train_images)).batch(1).shuffle(100).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_masks, test_images)).batch(1).prefetch(tf.data.AUTOTUNE)
    
    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator_fn)
    
    def compute_metrics(gen_output, target):
        """ Compute SSIM & PSNR for image quality assessment """
        ssim = tf.image.ssim(gen_output, target, max_val=1.0).numpy().mean()
        psnr = tf.image.psnr(gen_output, target, max_val=1.0).numpy().mean()
        return psnr, ssim
    
    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            if use_conditional_disc:
                disc_real_output = discriminator_fn([input_image, target], training=True)
                disc_generated_output = discriminator_fn([input_image, gen_output], training=True)
            else:
                disc_real_output = discriminator_fn(target, training=True)
                disc_generated_output = discriminator_fn(gen_output, training=True)

            
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_fn(disc_generated_output, gen_output, target, input_image)
            disc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output) + \
                        tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
            
        generator_optimizer.apply_gradients(zip(gen_tape.gradient(gen_total_loss, generator.trainable_variables), generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_tape.gradient(disc_loss, discriminator_fn.trainable_variables), discriminator_fn.trainable_variables))
        return gen_total_loss, disc_loss, gen_output, target
    
    progress_test_image, progress_test_target = next(iter(test_dataset.take(1)))
    progress_save_path = os.path.join(save_dir, "progress")

    for step, (input_image, target) in train_dataset.repeat().take(steps).enumerate():
        gen_loss, disc_loss, gen_output, target = train_step(input_image, target)

        # Compute SSIM and PSNR
        psnr, ssim = compute_metrics(gen_output, target)

        # Save loss to CSV
        with open(loss_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step+1, gen_loss.numpy(), disc_loss.numpy(), psnr, ssim])
        
        # ✅ Write losses to TensorBoard logs
        with summary_writer.as_default():
            tf.summary.scalar("Generator Loss", gen_loss, step=step+1)
            tf.summary.scalar("Discriminator Loss", disc_loss, step=step+1)
            tf.summary.scalar("PSNR", psnr, step=step+1)
            tf.summary.scalar("SSIM", ssim, step=step+1)

        if (step + 1) % 10000 == 0:  # Save progress every 10000 steps 
          print(f"Step {step+1}: Saving progress image...")
          generate_images(generator, progress_test_image, progress_test_target, progress_save_path, step+1, apply_noise, prefix="step")
          
        print(f"Step {step+1}: Gen Loss = {gen_loss.numpy():.4f}, Disc Loss = {disc_loss.numpy():.4f}")
        
        if (step + 1) % 50000 == 0:
            print(f"Step {step+1}: Saving checkpoint...")
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))

    # ✅ Generate Predictions for the Entire Test Set
    test_predictions_path = os.path.join(save_dir, "test_predictions")
    os.makedirs(test_predictions_path, exist_ok=True)

    print("Generating final test predictions...")
    for idx, (test_input, tar) in enumerate(test_dataset):
        generate_images(generator, test_input, tar, test_predictions_path, idx, apply_noise, prefix="test")

    # Save final models
    generator.save(os.path.join(save_dir, "generator.h5"))
    discriminator_fn.save(os.path.join(save_dir, "discriminator.h5"))
    print("Training complete. Models saved.")

