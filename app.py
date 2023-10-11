from flask import Flask, request, jsonify
from keras.models import load_model
import tensorflow as tf
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io
from decouple import config

app = Flask(__name__)

@app.route('/result', methods=['GET'])
def process_image():
    
    client = MongoClient('mongodb+srv://niravjangale:GhostUchiha@cluster0.oobzqch.mongodb.net/?retryWrites=true&w=majority')
    db = client['test']
    collection = db['images']
    model_file_path = "https://drive.google.com/file/d/1sc1nqI3m_ncFLINkQ4hhqqDJD2LRJ4qa/view?usp=sharing"  # Update with the correct path
    model = load_model(model_file_path)

    latest_image = collection.find_one(sort=[('_id', -1)])
    image_data = latest_image['img']['data']
    img = Image.open(io.BytesIO(image_data))

    try:
        img = img.resize((224, 224))
        img_tensor = tf.keras.utils.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255

        pred = model.predict(img_tensor)
        result = pred[0][0] * 100

        if result:
            return jsonify({'result': result})
        else:
            return jsonify({'message': 'Congratulations, You Are Safe'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
