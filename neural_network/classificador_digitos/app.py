from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)
model = load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        print("Nenhuma imagem recebida.")
        return jsonify({'error': 'Imagem não enviada'}), 400

    print("Imagem recebida.")

    img_data = base64.b64decode(data['image'].split(",")[1])
    image = Image.open(io.BytesIO(img_data)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    result = int(np.argmax(prediction))

    print(f"Número previsto: {result}")
    return jsonify({'digit': result})

if __name__ == '__main__':
    app.run(debug=True)