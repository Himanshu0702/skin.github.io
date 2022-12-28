import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras_preprocessing.image import load_img
from flask import Flask, render_template, request


app = Flask(__name__)

new_model = keras.models.load_model('.\models\mdl_ver_1.h5', compile=False)


@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['image_file']
    image_path = "./images/" + image_file.filename
    image_file.save(image_path)

    image = load_img(image_path)
    tf_image = np.array(image)
    resize = tf.image.resize(tf_image, (128, 128))
    yhat = new_model.predict(np.expand_dims(resize/1, 0))

    yhat = list(yhat)

    return render_template('index.html', prediction=yhat)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
