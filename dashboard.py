import streamlit as st
import pandas as pd
from model import hyperparameter_tuning ,conf_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from model import load_dataset,knn,set_kvalue
import sqlite3
from sklearn.metrics import roc_curve,auc
# st.title("hello streamlit")
# st.header("header")
# st.subheader("subheader")
# st.text("this is text")

# df = pd.DataFrame({
#     "age":[1,2,34,34,5,3,34,45],
#     "salary":[34435,43523,346346,346346,345345,2342454,676443,3456667],
#     "purchased":[1,0,1,0,1,0,1,0,]
# })
# st.dataframe(df,height=390,width=500)
# st.json({
#     "age":[1,2,34,34,5,3,34,45],
#     "salary":[34435,43523,346346,346346,345345,2342454,676443,3456667],
#     "purchased":[1,0,1,0,1,0,1,0,]
# })

# st.line_chart({
#     "age":[1,2,34,34,5,3,34,45],
#     "salary":[34435,43523,346346,346346,345345,2342454,676443,3456667],
#     "purchased":[1,0,1,0,1,0,1,0,]
# },)



nav =st.sidebar.radio("Navigation",["Home","Insert","Predict","Data Analysis","Hypertuning"])
if nav == "Home" :
    st.title("Welcome to Real time Data Analysis and Prediction of car purchase DashBoard")
    st.write("Here You Will Get What you need from your data")
    st.image("car-image.jpg")
    

if nav == "Predict":
    st.title("Predict and Analyze whether the customer will buy car or not")
    st.write(" ")
    age = st.number_input('Enter Age of Customer')
    st.write(" ")
    estimatsalary = st.number_input('Enter EstimatedSalary of Customer')
    st.write(" ")
    if st.button("predict"):
        if age > 17 and estimatsalary >1000:
            prediction, nearest_neighbors,probability = knn(age,estimatsalary)
            st.write("Prediction : ",prediction)
            st.write("Probability of Purchasing Car  : ",probability*100,"%")
            st.table(nearest_neighbors)

if nav =="Insert":
    st.title("Insert New Customer Data into DataBase")
    st.write("")
    age = st.number_input("Enter Age")
    st.write(" ")
    estimatsalary = st.number_input("Enter EstimatedSalary")
    st.write(" ")
    purchase = None
    
    bought = st.radio("Has Customer bought the car",("Yes","No"))
    if bought == "Yes":
        purchase = 1
    if bought =="No":
        purchase = 0
    
    if st.button("Submit"):
        conn = sqlite3.connect('car_prediction.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO customers (Age, EstimatedSalary, Purchased) VALUES (?, ?, ?)', (age, estimatsalary, purchase))
        conn.commit()
        st.write("Inserted Customer Data into DataBase Successfully ")
        
    




if nav=="Data Analysis":
    st.title("Customer Data Analysis")
    st.write("summary of numerical data")
    st.write(" ")
    df = pd.DataFrame()
    X,y = load_dataset()
    df["Age"] = [i[0] for i in X]
    df["EstimatedSalary"] = [i[1] for i in X]
    df["Purchased"] = y

    st.table(df.describe())
    df["Purchased"] = ["yes" if i==1 else 'no' for i in y]
    st.write(" ")
    st.text("Distribution of age in Customer Dataset")
    fig = px.histogram(df, x="Age",nbins=10)
    st.plotly_chart(fig)
    st.write(" ")
    st.write("Customer Age Vs Purchase Plot")
    fig = px.scatter(df, x="Age", y="Purchased",color="Purchased",color_discrete_map={"yes": "blue","no": "red"})
    st.plotly_chart(fig)
    st.write(" ")

    st.text("Distribution of EstimatedSalary in Customer Dataset")
    fig = px.histogram(df, x="EstimatedSalary")
    st.plotly_chart(fig)
    st.write(" ")

    st.write("Customer EstimatedSalary Vs Purchase Plot")
    fig = px.scatter(df, x="EstimatedSalary", y="Purchased",color="Purchased",color_discrete_map={"yes": "blue","no": "red"})
    st.plotly_chart(fig)
    st.write(" ")
    st.text("Distribution of Purchase in Customer Dataset")
    fig = px.histogram(df, x="Purchased",color="Purchased")
    st.plotly_chart(fig)
    st.write(" ")
    st.text("Scatter plot between Age and EstimatedSalary")
    
    fig = px.scatter(df, x="Age", y="EstimatedSalary",color="Purchased",color_discrete_map={"yes": "blue","no": "red"})
    st.plotly_chart(fig)


if nav == "Hypertuning":
    st.title("Find best parameters to improve the prediction")
    #  hypertuning graph
    def hyper_tuning():
        kvalue = st.number_input('Enter Max Kvalue')
        score = hyperparameter_tuning(k_value=int(kvalue),p=1)
        fig = plt.figure(figsize=(15,8))

        df  = pd.DataFrame(score)
        fig = go.Figure()
        fig = go.Figure(data=go.Scatter(x=df["kval"], y=df["accuracy"],mode='lines+markers', line_color='#ffe476',name="accurcy"))
        fig.add_trace(go.Scatter(x=df["kval"], y=df["precision"],mode='lines+markers', line_color='red',name="precision"))
        fig.add_trace(go.Scatter(x=df["kval"], y=df["recall"],mode='lines+markers', line_color='green',name="recall"))
        fig.add_trace(go.Scatter(x=df["kval"], y=df["f1_score"],mode='lines+markers', line_color='blue',name="f1_score"))


        plt.title(' K Value vs Score ')
        plt.xlabel('K-Values')
        plt.ylabel('Score')
        plt.xticks(fontsize=14)
        plt.legend()
        st.plotly_chart(fig)
        

    hyper_tuning()
    st.write(" ")
    def printconf_matrx():
        fig  = plt.figure(figsize=(3,3))

        kvalue = st.number_input('Enter  Kvalue')
        
        if kvalue>2:
            values=conf_matrix(int(kvalue))
            data = values['conf_matrix']
            y_test = values["y_test"]
            y_pred = values["yhat"]
            # fig = go.Figure(data=go.Heatmap(z=data,
            #                 text=data,
            #                 texttemplate="%{text}",))
            fig = px.imshow(data, text_auto=True, aspect="auto",labels=dict(x="Predicted", y="Actual"),x=["No","Yes"],y=["No","Yes"])
            st.plotly_chart(fig)
            fpr,tpr,thresold = roc_curve(y_test,y_pred)
            auc_curve = auc(fpr,tpr)
            fig = px.line(x=fpr, y=tpr,labels={"x":"FPR","y":"TPR"},title="ROC curve")
            
            st.plotly_chart(fig)
        kval = st.number_input("Set Best K value")
        st.write(" ")
        if st.button("set kvalue"):
            set_kvalue(kval=int(kval))
            st.write("kvalue is set to :",int(kval))
        
    st.text("Build confusion Matrix and Roc Curve")
    
    printconf_matrx()
