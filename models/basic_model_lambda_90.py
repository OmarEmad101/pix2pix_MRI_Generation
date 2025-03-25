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
    LAMBDA = 90

    # Weighted L1 loss calculation
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[128, 128, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 128, 128, 2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 64, 64, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 32, 32, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 16, 16, 256)
    down4 = downsample(512, 4)(down3)  # (batch_size, 8, 8, 512)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 10, 10, 512)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 7, 7, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 9, 9, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 6, 6, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

data_dir = os.environ.get("DATA_DIR")
if not data_dir:
    raise ValueError("Error: DATA_DIR is not set. Make sure to run the script from `main.py`.")

# Get the base directory (where the script is running from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define relative save path
save_dir = os.path.join(BASE_DIR, "..", "results", "basic_model_lambda_90")

if __name__ == "__main__":
    train_pix2pix(
        steps=100000,
        normalization=True,
        data_dir=data_dir,
        save_dir=save_dir,
        generator_loss_fn=generator_loss,
        discriminator_fn=discriminator,
        use_conditional_disc=True
    )
