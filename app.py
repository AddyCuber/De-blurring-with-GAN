import os
from flask import Flask, request, render_template, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import h5py

# Custom layer definition
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1], 
                input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor,
                     [[0,0], 
                      [padding_height, padding_height],
                      [padding_width, padding_width], 
                      [0,0]],
                     'REFLECT')

def Generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # 7x7 Conv n64
    x = ReflectionPadding2D(padding=(3,3))(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=(7,7), padding='valid',
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 3x3 Conv n128s2
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', strides=2,
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 3x3 Conv n256s2
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', strides=2,
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 9 resnet blocks n256
    for _ in range(9):
        x = res_net(x, 256, kernel_size=(3,3), apply_dropout=True)

    # unsampling n128
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same',
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # unsampling n64
    x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same',
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # 7x7 Conv n64
    x = ReflectionPadding2D(padding=(3,3))(x)
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(7,7), padding='valid',
               kernel_initializer=initializer)(x)
    x = tf.nn.tanh(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def res_net(input, filters, kernel_size=(3,3), apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    x = ReflectionPadding2D(padding=(1,1))(input)
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='valid',
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = ReflectionPadding2D(padding=(1,1))(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='valid',
               kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if apply_dropout:
        x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Add()([x, input])
    return x

app = Flask(__name__)

# Create and load the model
with tf.keras.utils.custom_object_scope({'ReflectionPadding2D': ReflectionPadding2D}):
    model = Generator()
    model.load_weights('deblur_generator.h5')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image):
    # Convert image to RGB mode if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original size for later use
    original_size = image.size
    
    # Calculate scaling factor to fit within 256x256 while maintaining aspect ratio
    scale = min(256 / original_size[0], 256 / original_size[1])
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    
    # Resize image maintaining aspect ratio
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Create a new square image with padding
    square_image = Image.new('RGB', (256, 256), (0, 0, 0))
    
    # Calculate position to paste the resized image
    paste_x = (256 - new_size[0]) // 2
    paste_y = (256 - new_size[1]) // 2
    
    # Paste the resized image onto the square image
    square_image.paste(image, (paste_x, paste_y))
    
    # Convert to numpy array and normalize
    img_array = np.array(square_image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_size

def postprocess_image(predicted, original_size):
    # Remove batch dimension
    predicted = predicted[0]
    # Denormalize
    predicted = (predicted * 255).astype(np.uint8)
    # Convert to PIL Image
    image = Image.fromarray(predicted)
    
    # Calculate scaling factor to fit within 256x256 while maintaining aspect ratio
    scale = min(256 / original_size[0], 256 / original_size[1])
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    
    # Calculate position to crop
    crop_x = (256 - new_size[0]) // 2
    crop_y = (256 - new_size[1]) // 2
    
    # Crop the image to the original aspect ratio
    image = image.crop((crop_x, crop_y, crop_x + new_size[0], crop_y + new_size[1]))
    
    # Resize back to original size
    image = image.resize(original_size, Image.Resampling.LANCZOS)
    
    return image

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/deblur', methods=['POST'])
def deblur():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    
    # Read and preprocess the image
    input_image = Image.open(file.stream)
    processed_image, original_size = preprocess_image(input_image)
    
    # Generate deblurred image
    deblurred = model.predict(processed_image)
    deblurred_image = postprocess_image(deblurred, original_size)
    
    # Save both input and output images
    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    output_path = os.path.join(UPLOAD_FOLDER, 'output.jpg')
    
    # Convert input image to RGB mode before saving
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    input_image.save(input_path)
    deblurred_image.save(output_path)
    
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True) 