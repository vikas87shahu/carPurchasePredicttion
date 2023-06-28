from flask import Flask, render_template,jsonify, request
app = Flask(__name__)

import sqlite3

## GLOBAL VARS
prediction = 0
nearest_customers = []
chance_of_buying = 0


"""Rough KNN Model"""
from model import knn
from model import hyperparameter_tuning
from model import conf_matrix


"""## Flask"""

@app.route('/insert',methods = ['GET'])
def insert_data():
    cust_data = request.get_json()
    print(cust_data)
    age = cust_data['age']
    salary = cust_data['salary']
    purchase = cust_data['purchase']
    conn = sqlite3.connect('car_prediction.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO customers (Age, EstimatedSalary, Purchased) VALUES (?, ?, ?)', (age, salary, purchase))
    conn.commit()
    return jsonify({'age':age,'salary':salary,'purchase':purchase,'successfully inserted':'yes'})


@app.route('/input',methods = ['POST'])
def input_data():
    global prediction
    global nearest_customers
    global chance_of_buying 
    cust_data = request.get_json()
    age = cust_data['age']
    salary = cust_data['salary']
    prediction, nearest_customers, chance_of_buying = knn(age, salary)
    return jsonify({'prediction':prediction,'nearest_customers':nearest_customers,'chance_of_buying':chance_of_buying})



@app.route('/hyper_tune',methods=['POST'])
def hyper_tuning():
    kvalue1 = request.get_json()['kvalue']
    score = hyperparameter_tuning(k_value=kvalue1,p=1)
    
    return jsonify({"score":score})

@app.route("/conf_matrix",methods=['POST'])
def getconf_matrix():
    kvalue1 = request.get_json()['kvalue']
    matrix = conf_matrix(kvalue1)
    

    print(matrix)
    return jsonify({"confusion_matrix":matrix})

if __name__ == '__main__':
    app.run(debug = True, port = 5001)