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
     return 'NIL'

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

def min_pred_water(value):
 dataset  = pd.read_csv('Mineral_.csv')
 x = dataset.iloc[:,4].values

 y = dataset.iloc[:,0].values
 print(x)
 print(y)
 x= x.reshape(-1, 1)
 y= y.reshape(-1, 1)
 min=np.min(x)
 max= np.max(x)
 min_e = min-2
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
 min_e = min-2
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

def min_estimate(p,q,r,data1):
 import numpy as np
 import pandas as pd
 from sklearn.model_selection import train_test_split
 from sklearn.svm import LinearSVR
 from sklearn.multioutput import MultiOutputRegressor
 
 dataset=pd.read_csv("data1.csv")
 X1=dataset.iloc[:,11].values
 X2=dataset.iloc[:,14].values
 X3=dataset.iloc[:,15].values
 X=[]
 for i in range (0,len(X1)):
   x=[]
   x.append(X1[i])
   x.append(X2[i])
   x.append(X3[i])
   X.append(x)


 Y=[]
 y1=dataset.iloc[:,6].values
 y2=dataset.iloc[:,7].values
 y3=dataset.iloc[:,9].values
 # y4=dataset.iloc[:,9].values

 for i in range (0,len(X)):
   y=[]
   y.append(y1[i])
   y.append(y2[i])
   y.append(y3[i])
   # y.append(y4[i])
   Y.append(y)

 # X=scaler.fit_transform(X)
 # Y=scaler.fit_transform(Y)
 x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.9,random_state=0)
 x_train=np.array(x_train)
 y_train=np.array(y_train)
 x_test=np.array(x_test)

 x_t=X[0:8]
 y_t=Y[0:8]


 model  = LinearSVR()
 # define the chained multioutput wrapper model
 wrapper =  MultiOutputRegressor(model)
 # fit the model on the whole dataset
 wrapper.fit(x_t, y_t)
 y_p1=wrapper.predict([[p,q,r]])
 data1["Calcium"]=y_p1[0][0]
 data1["Magnesium"]=y_p1[0][1]
 data1["Sodium"]=y_p1[0][2]
 return data1
 
 


@app.route('/estimate',methods=['POST','GET'])
def post_estimate():
 data3  =request.get_json()
 reflected=data3["reflected"]
 moisture=data3["moisture"]
 infrared=data3["infrared"]
 data4= min_estimate(reflected,infrared,moisture,data3)
 return data4
 
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

@app.route('/predict_water',methods=['POST','GET'])
def post_water():
 data3  =request.get_json()
 value=data3["data_i"]
 data3["min_name"] = min_pred_water(value)
 return data3


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
