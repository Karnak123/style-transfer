"""file containing utility functions required by main program"""

# imports
import tensorflow as tf
import numpy as np
import PIL.Image


# Utility functions


def load_img(path_to_img):
    """
    :param path_to_img: filepath to downloaded image
    :return img: tf.image
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def tensor_to_image(tensor):
    """
    :param tensor: EagerTensor
    :return: PIL.Image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def get_images(content, style):
    """
    :param content: url to content image
    :param style: url to style image
    :return: filepath to content and style images
    """
    content_path = tf.keras.utils.get_file("content.jpg", content)
    style_path = tf.keras.utils.get_file("style.jpg", style)
    return content_path, style_path


# VGG


def vgg_layers(layer_names):
    """
    :param layer_names: list of layers to be extracted
    :return: tf.keras.Model containing only specified layers
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)


def gram_matrix(input_tensor):
    """
    :param input_tensor: intermediate tensor
    :return: style estimate
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


# Training


@tf.function()
def clip_0_1(image):
    """
    :param image: training intermediate tensor
    :return: clipped tensor in range [0, 1]
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
