from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('Api/model/wine.pkl')

@app.route('/')
def home():
    return "Wine", 200
@app.route('/wine/classify')
def wineClassify():    
    features = request.get_json()
    test = np.array([[
        features.get('fixed_acidity'),
        features.get('volatile_acidity'),
        features.get('citric_acid'),
        features.get('residual_sugar'),
        features.get('chlorides'),
        features.get('free_sulfur_dioxide'),
        features.get('total_sulfur_dioxide'),
        features.get('density'),
        features.get('pH'),
        features.get('sulphates'),
        features.get('alcohol'),
        features.get('quality')
    ]])
    label = int(model.predict(test)[0])
    if label == 0:
        labelName = 'red'
    else:
        labelName = 'white'
    return jsonify({'label': label, 'name': labelName}), 200

if __name__=='__main__':
    app.run(debug=True)