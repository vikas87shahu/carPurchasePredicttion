import sqlite3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,f1_score)

conn = sqlite3.connect('car_prediction.db')
cursor = conn.cursor()

# GLOBAL DATASET
dataset = []
labels = []
kvalue_g=20

def load_dataset():
    global dataset
    global labels
    dataset =[]
    labels=[]
    ###### another thread for parallel operation (since flask performs parallel operations)
    conn = sqlite3.connect('car_prediction.db')
    cursor = conn.cursor()
    cursor.execute('SELECT Age, EstimatedSalary, Purchased FROM customers')
    rows = cursor.fetchall()
    for row in rows:
        dataset.append([row[0], row[1]])
        labels.append(row[2])
    return dataset, labels

def one_counter(nearest_neighbors):
    count = 0
    for l in nearest_neighbors:
        if l[2]: count+=1
    return count


def hyperparameter_tuning(k_value = None,p = 1):
    if k_value is None:
        k_value = 40
    if p is None :
        p = 2
    
    X,y =load_dataset()
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train= scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    score = {"kval":[],'accuracy':[],'precision':[],'recall':[],'f1_score':[]}

    print(k_value,p)
    # Will take some time
    for i in range(1,k_value):
        neigh = KNeighborsClassifier(n_neighbors = i,p=p).fit(X_train,y_train)
        yhat = neigh.predict(X_test)
        score["kval"].append(i)
        score['accuracy'].append(round(accuracy_score(y_test, yhat),3))
        score['precision'].append(round(precision_score(y_test, yhat),3))
        score['recall'].append(round(recall_score(y_test, yhat),3))
        score['f1_score'].append(round(f1_score(y_test, yhat),3))
    return score

def conf_matrix(kvalue):
    X,y =load_dataset()
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train= scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    neigh = KNeighborsClassifier(n_neighbors = kvalue).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    conf_mat = []
    for i in [*confusion_matrix(y_test,yhat)]:
        conf_mat.append([int(k)for k in i])
    
    matr = dict({"Kvlaue":kvalue,"conf_matrix":conf_mat,"y_test":y_test,"yhat":yhat})
        
    return matr

def set_kvalue(kval):
    global kvalue_g
    if kval>2:
        kvalue_g = kval
    else :
        kvalue_g = 20

def train_model():
    dataset, labels = load_dataset()
    model = KNeighborsClassifier(n_neighbors=24)
    scaler = StandardScaler()
    scaler.fit(dataset)
    print(len(dataset))
    scaled_data= scaler.transform(dataset)
    model.fit(scaled_data, labels)
    return model,scaler

def get_nearest_neighbors(model, data_point, k):
    distances, indices = model.kneighbors(data_point, n_neighbors=k)
    return distances, indices

def knn(age, salary):
    global kvalue_g
    model,scaler = train_model()
    data_point = scaler.transform([[age, salary]])
    prediction = int(model.predict(data_point))
    distances, indices = get_nearest_neighbors(model, data_point, k=kvalue_g)

    nearest_neighbors = []
    for i in indices[0]:
        neighbor = dataset[i]+[int(labels[i])]  # Assuming 'dataset' is accessible here
        nearest_neighbors.append(neighbor)

    return (prediction, nearest_neighbors, one_counter(nearest_neighbors)/len(nearest_neighbors))

