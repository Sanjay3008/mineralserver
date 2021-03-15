import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template

app = Flask(__name__)


def min_pred(value):
 dataset = pd.read_csv('Mineral.csv')
 x = dataset.iloc[:, :-1]
 x = np.int32(np.around(x * 10 ** 4))
 value=str((int)(value)*10000)
 y = dataset.iloc[:, -1]

 min = np.min(x)
 max = np.max(x)
 min_e = min - 4
 max_e = max + 4
 if (((int)(value) < min_e) | ((int)(value) > max_e)):
     return 'No_Element'

 from sklearn.preprocessing import LabelEncoder
 encoder = LabelEncoder()
 y = encoder.fit_transform(y)

 from sklearn.model_selection import train_test_split
 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

 from sklearn.tree import DecisionTreeClassifier
 classifier = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=1000)
 classifier.fit(x, y)

 y_p = classifier.predict([[value]])
 mineral_name = encoder.inverse_transform(y_p)
 return str(mineral_name)


def min_pred_air(value):
 dataset  = pd.read_csv('Mineral_.csv')
 x = dataset.iloc[:,2].values

 y = dataset.iloc[:,0].values
 print(x)
 print(y)
 x= x.reshape(-1, 1)
 y= y.reshape(-1, 1)
 min=np.min(x)
 max= np.max(x)
 min_e = min-2;
 max_e = max+2
 
 from sklearn.preprocessing import LabelEncoder
 label_encoder  = LabelEncoder()
 y= label_encoder.fit_transform(y)
 
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.55,random_state=0)
 
 from sklearn.tree import DecisionTreeClassifier
 classifier=DecisionTreeClassifier()
 classifier.fit(x,y)
 
 y_p = classifier.predict([[value]])
 mineral_name = label_encoder.inverse_transform(y_p)
 return str(mineral_name)


 
@app.route('/post',methods=['POST','GET'])
def post():
 data1  =request.get_json()
 value=data1["data_i"]
 data1["min_name"] = min_pred(value)
 return data1

@app.route('/predict_air',methods=['POST','GET'])
def post_air():
 data2  =request.get_json()
 value=data2["data_i"]
 data2["min_name"] = min_pred_air(value)
 return data2


@app.route('/connect',methods=['POST','GET'])
def handle_request():
    return "Successful Connection"
 
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
