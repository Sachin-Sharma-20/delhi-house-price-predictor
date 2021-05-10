from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import time

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


def preprocess(df):
    df = df.copy()
    df = df.reindex(sorted(df.columns), axis=1)
    loca = df['Locality']
    dc = []
    for i in loca:
        st = str(i)
        dc.append(abs(hash(st)))
    df['Locality'] = list(dc) 
    df['Furnishing'] = df['Furnishing'].replace({
        'Semi-Furnished': 1,
        'Unfurnished': 0,
        'Furnished': 2
    })
    df['Type'] = df['Type'].replace({'Builder_Floor': 1, 'Apartment': 0})
    df['Status'] = df['Status'].replace({
        'Ready_to_move': 1,
        'Almost_ready': 0
    })
    df['Transaction'] = df['Transaction'].replace({
        'Resale': 0,
        'New_Property': 1
    })
    for column in [
            'Transaction', 'Status', 'Type', 'Furnishing', 'Bathroom',
            'Parking'
    ]:
        df[column] = df[column].fillna(df[column].mean())
    return df


@app.route('/predict',methods=['POST'])
def predict():
    Area = request.form['area']
    BHK = request.form['bhk']
    Bathroom = request.form['bathroom']
    Furnishing = request.form['furnishing']
    Locality = request.form['locality']
    Parking = request.form['parking']
    Status = request.form['status']
    Transaction = request.form['transaction']
    Type = request.form['type']

    data_for_pred = {
        'Area': float(Area),
        'BHK': int(BHK),
        'Bathroom': float(Bathroom),
        'Furnishing': Furnishing,
        'Locality': Locality,
        'Parking': float(Parking),
        'Status': Status,
        'Transaction': Transaction,
        'Type': Type,
    }
    df2 = pd.DataFrame(data_for_pred, index=[0])
    df2 = preprocess(df2)
    pred = model.predict(df2)
    output = str(int(round(pred[0])))
    output = [int(x) for x in output]
    print((output))
    n = len(output)
    if (n>3):
        k = n-3
        while(k>0):            
            output.insert(k,',')
            k-=2
    output = ''.join(str(x) for x in output)
    print((output))
    return render_template('home.html', prediction_text=f"The Estimated Price is ₹ {output}") 


if __name__ == '__main__':
    app.run(debug=True)