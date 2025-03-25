import tensorflow as tf
import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_utils import train_pix2pix
from train_utils import downsample


def generator_loss(disc_generated_output, gen_output, target, input_mask):
   # Original GAN loss component
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    LAMBDA = 100

    # Create weight matrix for different regions
    weights = tf.where(input_mask == 3.0, 25.0, 1.0)  # Infarction (25x)
    weights = tf.where(input_mask == 4.0, 100.0, weights)  # No-reflow (100x)
    weights = tf.where(input_mask == 2.0, 12.0, weights)  # Normal myocardium (12x)
    weights = tf.where(input_mask == 1.0, 7.0, weights)  # Cavity (7x)

    # Expand weights to match image dimensions [batch, H, W, 1]
    weights = tf.expand_dims(weights, axis=-1)

    # Weighted L1 loss calculation
    l1_loss = tf.reduce_mean(weights * tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    # Define the discriminator model here
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

data_dir = os.environ.get("DATA_DIR")
if not data_dir:
    raise ValueError("Error: DATA_DIR is not set. Make sure to run the script from `main.py`.")

# Get the base directory (where the script is running from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define relative save path
save_dir = os.path.join(BASE_DIR, "..", "results", "basic_model_weighted_loss_lambda_100")

if __name__ == "__main__":
    train_pix2pix(
        steps=150000,
        normalization=True,
        data_dir=data_dir,
        save_dir=save_dir,
        generator_loss_fn=generator_loss,
        discriminator_fn=discriminator,
        use_conditional_disc=True
    )
