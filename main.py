"""main program to perform style transfer"""

# imports
import argparse
import os

import tensorflow as tf

import util

# setup environment
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
tf.autograph.set_verbosity(3)

# parse input
parser = argparse.ArgumentParser()
parser.add_argument(
    "-content",
    type=str,
    default="https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
)
parser.add_argument(
    "-style",
    type=str,
    default="https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
)
parser.add_argument("-epochs", type=int, default=20)
args = parser.parse_args()

# get images from url
content_path, style_path = util.get_images(args.content, args.style)

# load image
content_image = util.load_img(content_path)
style_image = util.load_img(style_path)

# VGG
content_layers = ["block5_conv2"]
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

NUM_CONTENT_LAYERS = len(content_layers)
NUM_STYLE_LAYERS = len(style_layers)


# Build model
class StyleContentModel(tf.keras.models.Model):
    """
    When called on an image returns style and content tensors
    """

    def __init__(self, s_layers, c_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = util.vgg_layers(s_layers + c_layers)
        self.style_layers = s_layers
        self.content_layers = c_layers
        self.num_style_layers = len(s_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """
        :param inputs: eager tensor
        :return gram matrix of style layers and content of content layers
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = list(map(util.gram_matrix, style_outputs))

        content_dict = dict(zip(self.content_layers, content_outputs))

        style_dict = dict(zip(self.style_layers, style_outputs))

        return {"content": content_dict, "style": style_dict}


extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)["style"]
content_targets = extractor(content_image)["content"]

image = tf.Variable(content_image)

epochs = args.epochs
STEPS_PER_EPOCH = 100

# change values for experimentation
STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4
TOTAL_VARIATION_WEIGHT = 30


def style_content_loss(outputs):
    """
    :param outputs: output of training step
    :return: net loss
    """
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= STYLE_WEIGHT / NUM_STYLE_LAYERS

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= CONTENT_WEIGHT / NUM_CONTENT_LAYERS
    loss = style_loss + content_loss
    return loss


def train_step(image):
    """:param image: tensor containing target image"""
    with tf.GradientTape() as tape:
        tape.watch(image)
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += TOTAL_VARIATION_WEIGHT * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    opt.apply_gradients([(grad, image)])
    image.assign(util.clip_0_1(image))
    del tape


# training loop
for n in range(epochs):
    for m in range(STEPS_PER_EPOCH):
        train_step(image)

# save file to disk
FILE_NAME = "stylized-image.png"
util.tensor_to_image(image).save(FILE_NAME)
