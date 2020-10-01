# style-transfer

This is a basic implementation of Neural Style Transfer with VGG 19 layers using TensorFlow 2.3.0
morphed into command line tool as demonstrated below. The code is separated into util.py
and main.py where util.py contains functions definitions and main.py contains the logical
sequence of code.

## Demo
`python main.py`

The code runs the program on default configuration and saves generated image as "stylized-image.py". 
The default number of epochs is 20, however around 10 epochs are sufficient and further epochs are not 
of much effect in most cases.

## Usage
`python main.py -content content_url -style style_url -epochs n`
where content_url is url to content image, style_url is url to style image and n is number of epochs,
all the arguments are optional. A GPU enabled system is highly recommended to run the program.