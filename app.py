from model.network import Net
from flask import Flask, jsonify, render_template, request
from preprocessing import *

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/digit_prediction', methods=['POST'])
def digit_prediction():
    if(request.method == "POST"):
        img = request.get_json()
        img = preprocess(img)
        print("preprocessing done......")
        net = Net()
        digit, probability = net.predict_with_pretrained_weights(img, 'weights.pkl')
        print(digit, probability)
        data = { "digit": str(digit), "probability": str(round(probability*100, 2))}
        return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
