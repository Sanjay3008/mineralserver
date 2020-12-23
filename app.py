import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template

app = Flask(__name__)


@app.route('/post',methods=['POST','GET'])
def post():
 data1  =request.get_json()
 return data1["data_i"]


 
@app.route('/predict',methods=['POST','GET'])
def predict():
 
    value=32.175
    dataset = pd.read_csv('Mineral.csv')
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    print(x)
    print(y)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    print(y)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=1000)
    classifier.fit(x_train, y_train)

    y_p = classifier.predict([[value]])
    mineral_name = encoder.inverse_transform(y_p)
    print(mineral_name)
    return str(mineral_name)

if __name__=="__main__":
    app.run()
