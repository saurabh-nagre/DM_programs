import math
from collections import Counter
from math import *
import random
import pandas as pd
import tkinter  as tk 
from tkinter import *
import numpy as np
from tkinter import ttk
from scipy.stats import norm
from numpy import mean
from numpy import std
from tkinter import filedialog
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_text
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


my_w = tk.Tk()
my_w.geometry("800x500")  
my_w.title('DATA VISUALIZE')

my_font1=('times', 12, 'bold')
l1 = tk.Label(my_w,width=40, text='Select & Read File',font=my_font1)  
l1.grid(row=1)
b1 = tk.Button(my_w, text='Browse File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2)

def euclidean_distance(x_test, x_train):
    distance = 0
    for i in range(len(x_test)-1):
        distance += (x_test[i]-x_train[i])**2
    return sqrt(distance)

def get_neighbors(x_test, x_train, num_neighbors):
    distances = []
    data = []
    for i in x_train:
        distances.append(euclidean_distance(x_test,i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()             #argsort() function returns indices by sorting distances data in ascending order
    data = data[sort_indexes]                      #modifying our data based on sorted indices, so that we can get the nearest neightbours
    return data[:num_neighbors]           

def prediction(x_test, x_train, num_neighbors):
    classes = []
    neighbors = get_neighbors(x_test, x_train, num_neighbors)
    for i in neighbors:
        classes.append(i[-1])
    predicted = max(classes, key=classes.count)              #taking the most repeated class
    return predicted

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def J(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def gradientdescent(X, y, lmd, alpha, num_iter, print_cost):

    # select initial values zero
    theta = np.zeros(X.shape[1])
    
    costs = []  
    
    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        # adding regularization 
        reg = lmd / y.size * theta
        # first theta is intercept
        # it is not regularized
        reg[0] = 0
        cost = J(h, y)
        
        gradient = np.dot(X.T, (h - y)) / y.size + reg
        theta = theta - alpha * gradient
    
        if print_cost and i % 100 == 0: 
            print('Number of Iterations: ', i, 'Cost : ', cost, 'Theta: ', theta)
        if i % 100 == 0:
            costs.append(cost)
      
    return theta, costs

def predict(X_test, theta):
    z = np.dot(X_test, theta)
    return sigmoid(z)

def logistic(X_train, y_train, X_test, lmd=0, alpha=0.1, num_iter=30000, print_cost = False):
    # Adding intercept
    intercept = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((intercept, X_train), axis=1)
    
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1)

    # one vs rest
    u=set(y_train)
    t=[]
    allCosts=[]   
    for c in u:
        # set the labels to 0 and 1
        ynew = np.array(y_train == c, dtype = int)
        theta_onevsrest, costs_onevsrest = gradientdescent(X_train, ynew, lmd, alpha, num_iter, print_cost)
        t.append(theta_onevsrest)
        
        # Save costs
        allCosts.append(costs_onevsrest)
        
    # Calculate probabilties
    pred_test = np.zeros((len(u),len(X_test)))
    for i in range(len(u)):
        pred_test[i,:] = predict(X_test,t[i])
    
    # Select max probability
    prediction_test = np.argmax(pred_test, axis=0)
    
    # Calculate probabilties
    pred_train = np.zeros((len(u),len(X_train)))
    for i in range(len(u)):
        pred_train[i,:] = predict(X_train,t[i])
    
    # Select max probability
    prediction_train = np.argmax(pred_train, axis=0)
    
    d = [allCosts,prediction_test, 
         prediction_train]
        
    return d
###########################################################################################
##################Naive Bayes functions####################################################
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def separate_by_class(dataset):
    separated = {}
    unique = np.unique(dataset.iloc[:,-1])
    # print(unique)
    for val in unique:
        separated[val] = []

    for i in range(len(dataset.iloc[:,-1])):
        separated[dataset.iloc[i,-1]].append(dataset.iloc[i,1:-1])
    # print(separated)
    return separated

def splitDataset(dataset, ratio):
    trainSize = int(len(dataset) * ratio)
    trainSet = []
    tempSet = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange((len(tempSet)))
        trainSet.append(tempSet.pop(index))
    return [trainSet, tempSet]

def fit_distribution(data):
    mu = mean(data)
    sigma = std(data)
    dist = norm(mu, sigma)
    return dist

def probability(X, prior, dist1, dist2, dist3, dist4):
    return prior * dist1.pdf(X.iloc[:,1]) * dist2.pdf(X.iloc[:,2]) * dist3.pdf(X.iloc[:,3]) * dist4.pdf(X.iloc[:,4])

def predictedClass(data, prob_setosa, prob_versicolor, prob_virginica):
    max_prob = [0] * len(data)
    predicted = [""] * len(data)
    for i in range(len(data)):
        max_prob[i] = max(prob_setosa[i], prob_versicolor[i], prob_virginica[i])
        if max_prob[i] == prob_setosa[i]:
            predicted[i] = "Iris-setosa"
        elif max_prob[i] == prob_versicolor[i]:
            predicted[i] = "Iris-versicolor"
        else:
            predicted[i] = "Iris-virginica"
    return predicted


 
# returns predictions for a set of examples
def getPredictions(info, test):
    predictions = []
    for i in range(len(test)):
        result = predict(info, test[i])
        predictions.append(result)
    return predictions
 
def accuracy(y_true, y_pred):
    num_correct = 0
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            num_correct+=1
    accuracy = num_correct/len(y_true)
    return accuracy
 
####################################Assingnment 3 - Decision Tree########################################################



def entropy(labels):
        entropy=0
        label_counts = Counter(labels)
        for label in label_counts:
            prob_of_label = label_counts[label] / len(labels)
            entropy -= prob_of_label * math.log2(prob_of_label)
        return entropy

def information_gain(starting_labels, split_labels):
        info_gain = entropy(starting_labels)
        ans=0
        for branched_subset in split_labels:
            ans+=len(branched_subset) * entropy(branched_subset) / len(starting_labels)
        info_gain-=ans
        return info_gain
def split(dataset, column):
        split_data = []
        col_vals = dataset[column].unique()
        for col_val in col_vals:
            split_data.append(dataset[dataset[column] == col_val])
        return(split_data)

def implementDT(data,attribute):
    window3 = tk.Tk()
    window3.geometry("1200x800")
    window3.title('Decision Tree')
    features = data.columns[1:]
    features = np.array(features)
    # print(features)
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:-1], data.iloc[:,0], test_size = 0.5)
    class_values = np.unique(y_train)
    y_pred = None
    r = None
    if attribute=='Information Gain':
        dt  = DecisionTreeClassifier(criterion='entropy',max_depth=2,random_state=1234)
        dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        fig = plt.figure(figsize=(4,3),dpi=150)
        tree.plot_tree(dt, feature_names=features,rounded=True, class_names=class_values,filled=True)
        r = export_text(dt)
        # print(y_pred)
    elif attribute=='Gain Ratio':
        dt  = DecisionTreeClassifier(max_depth=2,random_state=123)
        dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        fig = plt.figure(figsize=(4,3),dpi=150)
        tree.plot_tree(dt, feature_names=features,rounded=True, class_names=class_values,filled=True)
        r = export_text(dt)
        # print(y_pred)
    else:
        dt  = DecisionTreeClassifier(random_state=13,max_depth=2)
        dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        fig = plt.figure(figsize=(4,3),dpi=150)
        tree.plot_tree(dt, feature_names=features,rounded=True, class_names=class_values,filled=True)
        r = export_text(dt)
    test_df = X_test
    test_df['Actual'] = y_test
    test_df['Predicted'] = y_pred
    tv1 = ttk.Treeview(window3,height=10)
    tv1.grid(column=0,row=0,padx=5,pady=10,columnspan=10)
    tv1["column"] = list(test_df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)
        
    df_rows = test_df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)

    y_true = np.array(y_test)
    cfm = confusion_matrix(y_test, y_pred)
    acc = accuracy(y_true,y_pred)
    cm = tk.Text(window3)
    cm.grid(row=4,column=0)
    cm.insert(tk.END,'Confusion Matrix\n'+str(cfm)+'\n')    
    
    cm.insert(tk.END,'\na) Recognition rate: '+str(acc*100))
    cm.insert(tk.END,'\nb) Misclassification rate'+str((1-acc)*100)+"\n\n")


    cm.insert(tk.END,classification_report(y_test, y_pred,zero_division=0,))
    
    cm.insert(tk.END,"\n\n")
    cm.insert(tk.END,r)

    ac_label = tk.Label(window3,width=100, text='Accuracy : '+ str(acc),font=my_font1)
    ac_label.grid(row=2,column=0)
    plt.show()   
    window3.mainloop()

def calculate(data,classifier,size):
    window2 = tk.Tk()
    window2.geometry("800x500")  # Size of the window 
    
    Species = list(set(data['Species']))
    Specie1 = data[data['Species']==Species[0]]
    Specie2 = data[data['Species']==Species[1]]
    Specie3 = data[data['Species']==Species[2]]
    plt.scatter(Specie1['PetalLengthCm'], Specie1['PetalWidthCm'], label=Species[0])
    plt.scatter(Specie2['PetalLengthCm'], Specie2['PetalWidthCm'], label=Species[1])
    plt.scatter(Specie3['PetalLengthCm'], Specie3['PetalWidthCm'], label=Species[2])
    plt.xlabel('PetalLengthCM')
    plt.ylabel('PetalWidthCM')
    plt.legend()
    plt.title('Different Species Visualization')
    
    req_data = data.iloc[:,:]
    req_data.head(5)

    shuffle_index = np.random.permutation(req_data.shape[0])        #shuffling the row index of our dataset
    req_data = req_data.iloc[shuffle_index]

    tv1 = ttk.Treeview(window2,height=10)
    tv1.grid(column=0,row=0,padx=5,pady=10,columnspan=10)
    tv1["column"] = list(req_data.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)
        
    df_rows = req_data.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)

    train_size = int(req_data.shape[0]*0.01*size)

    train_df = req_data.iloc[:train_size,:] 
    test_df = req_data.iloc[train_size:,:]
    train = train_df.values
    test = test_df.values
    y_true = test[:,-1]

    print('Train_Shape: ',train_df.shape)
    print('Test_Shape: ',test_df.shape)
    X_train = train_df.iloc[:,1:-1]
    y_train = train_df.iloc[:,-1]
    x_test = test_df.iloc[:,1:-1]
    y_test = test_df.iloc[:,-1]

    plt.show()

    if classifier=='Logistic Regression classifier':
        logreg = LogisticRegression()
        window2.title("Logistic Regression")
        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(x_test)
        y_true = np.array(y_test)
        acc = accuracy(y_true,y_pred)
        ac_label = tk.Label(window2,width=100, text='Accuracy :'+str(acc),font=my_font1)
        ac_label.grid(row=1,column=0)

        for col in test_df.columns:
                if col=='Predicted_Species':
                    test_df.drop(['Predicted_Species'],axis=1)

        if 'Predicted_Species' not in test_df.columns:
                test_df.insert(6, 'Predicted_Species', y_pred, False)

        tv2 = ttk.Treeview(window2,height=10)
        for item in tv2.get_children():
                tv2.delete(item)
        tv2.grid(column=0,row=3,padx=5,pady=10,columnspan=10)
        tv2["column"] = list(test_df.columns)
        tv2["show"] = "headings"
        for column in tv2["columns"]:
                tv2.heading(column, text=column)

        df_rows = test_df.to_numpy().tolist()
        for row in df_rows:
                tv2.insert("", "end", values=row)

        cfm = confusion_matrix(y_test, y_pred)
        cm = tk.Text(window2)
        cm.grid(row=4,column=0)
        cm.insert(tk.END,str(cfm)+"\n")
        cm.insert(tk.END,'\na) Recognition rate: '+str(acc*100))
        cm.insert(tk.END,'\nb) Misclassification rate'+str((1-acc)*100)+"\n\n")
        cm.insert(tk.END,classification_report(y_test, y_pred))

    if classifier=='k-NN classifier':
        window2.title('KNN Classifier')
        
        def knnClassify(k):
            y_pred = []
            for i in test:
                y_pred.append(prediction(i, train, k))


            acc = accuracy(y_true, y_pred)

            ac_label = tk.Label(window2,width=100, text='Accuracy : '+ str(acc),font=my_font1)
            ac_label.grid(row=2,column=0)
            for col in test_df.columns:
                if col=='Predicted_Species':
                    test_df.drop(['Predicted_Species'],axis=1)

            if 'Predicted_Species' not in test_df.columns:
                test_df.insert(6, 'Predicted_Species', y_pred, False)

            tv2 = ttk.Treeview(window2,height=10)
            for item in tv2.get_children():
                tv2.delete(item)
            tv2.grid(column=0,row=5,padx=5,pady=10,columnspan=10)
            tv2["column"] = list(test_df.columns)
            tv2["show"] = "headings"
            for column in tv2["columns"]:
                tv2.heading(column, text=column)

            df_rows = test_df.to_numpy().tolist()
            for row in df_rows:
                tv2.insert("", "end", values=row)

            cfm = confusion_matrix(y_test,y_pred)
            cm = tk.Text(window2)
            cm.delete("1.0","end")
            cm.grid(row=6,column=0)
            cm.insert(tk.END,str(cfm))
            cm.insert(tk.END,'\na) Recognition rate: '+str(acc*100))
            cm.insert(tk.END,'\nb) Misclassification rate'+str((1-acc)*100)+"\n\n")
            cm.insert(tk.END,classification_report(y_test, y_pred))

        kLable = tk.Label(window2,width=50,font=my_font1,text='Select K Neighbour: ')
        kLable.grid(row=1,column=0)
        
        opt = [1,3,5,7]
        sel = StringVar(window2)
        sel.set(str(1))
        dd = tk.OptionMenu(window2,sel,*opt)
        dd.grid(column=1,row=1,padx=10,pady=10)
        
        btn = tk.Button(window2,text='Classify', 
            width=20,command = lambda:knnClassify(int(sel.get())))

        btn.grid(row=1,column=2)
        

        

    if classifier=="Naïve Bayesian Classifier":
        separated = separate_by_class(train_df)
        testX = test_df.iloc[:,1:-1]
        trainSet = train_df
        window2.title("Naïve Bayesian Classifier")
        testSet = test_df
        testSet = pd.DataFrame(testSet)
        test = test_df.values
        y_true = test[:,-1]
        actualTestClass = test_df.iloc[:,-1]
        actualTestClass.columns = ["0"]

        X_seto = separated['Iris-setosa']
        X_versi = separated['Iris-versicolor']
        X_virgi = separated['Iris-virginica']
        
        X_seto = pd.DataFrame(X_seto)
        X_versi = pd.DataFrame(X_versi)
        X_virgi = pd.DataFrame(X_virgi)

        prior_seto = len(X_seto) / len(train_df)
        prior_versi = len(X_versi) / len(train_df)
        prior_virgi = len(X_virgi) / len(train_df)

        X1_seto = fit_distribution(X_seto.iloc[:,0])
        X2_seto = fit_distribution(X_seto.iloc[:,1])
        X3_seto = fit_distribution(X_seto.iloc[:,2])
        X4_seto = fit_distribution(X_seto.iloc[:,3])
        
        X1_versi = fit_distribution(X_versi.iloc[:,0])
        X2_versi = fit_distribution(X_versi.iloc[:,1])
        X3_versi = fit_distribution(X_versi.iloc[:,2])
        X4_versi = fit_distribution(X_versi.iloc[:,3])

        X1_virgi = fit_distribution(X_virgi.iloc[:,0])
        X2_virgi = fit_distribution(X_virgi.iloc[:,1])
        X3_virgi = fit_distribution(X_virgi.iloc[:,2])
        X4_virgi = fit_distribution(X_virgi.iloc[:,3])
        

        # Calculate probability of each class for all testing dataset
        prob_seto = probability(testSet, prior_seto, X1_seto, X2_seto, X3_seto, X4_seto)
        prob_versi = probability(testSet, prior_versi, X1_versi, X2_versi, X3_versi, X4_versi)
        prob_virgi = probability(testSet, prior_virgi, X1_virgi, X2_virgi, X3_virgi, X4_virgi)

        # Predicted class is the class with highest probability
        y_pred = predictedClass(testSet, prob_seto, prob_versi, prob_virgi)
        
        # Evaluate model by calculate accuracy %
        acc = accuracy(y_true, y_pred)
        # print('Implemented Naive Bayes Accuracy = {0:.3f} %'.format(acc))

        cm = confusion_matrix(y_test, y_pred)
        for col in test_df.columns:
                if col=='Predicted_Species':
                    test_df.drop(['Predicted_Species'],axis=1)

        if 'Predicted_Species' not in test_df.columns:
                test_df.insert(6, 'Predicted_Species', y_pred, False)

        tv2 = ttk.Treeview(window2,height=10)
        for item in tv2.get_children():
                tv2.delete(item)
        tv2.grid(column=0,row=5,padx=5,pady=10,columnspan=10)
        tv2["column"] = list(test_df.columns)
        tv2["show"] = "headings"
        for column in tv2["columns"]:
                tv2.heading(column, text=column)

        df_rows = test_df.to_numpy().tolist()
        for row in df_rows:
                tv2.insert("", "end", values=row)

        cfm = confusion_matrix(y_test,y_pred)
        cm = tk.Text(window2)
        cm.delete("1.0","end")
        cm.grid(row=6,column=0)
        cm.insert(tk.END,cfm)
        cm.insert(tk.END,'\na) Recognition rate: '+str(acc*100))
        cm.insert(tk.END,'\nb) Misclassification rate'+str((1-acc)*100)+"\n\n")
        cm.insert(tk.END,classification_report(y_test, y_pred))
        
        ac_label = tk.Label(window2,width=100, text='Accuracy : '+ str(acc),font=my_font1)
        ac_label.grid(row=2,column=0)

    if classifier=="Three layer Artificial Neural Network (ANN) classifier":
        df_norm = data[data.columns[1:-1]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        unique = np.unique(data.iloc[:,-1])
        
        target = data.iloc[:,-1].replace(unique,range(len(unique)))
        df = pd.concat([df_norm, target], axis=1)
        train_test_per = (100-size)/100.0
        df['train'] = np.random.rand(len(df)) < train_test_per
        train = df[df.train == 1]
        train = train.drop('train', axis=1).sample(frac=1)
        test = df[df.train == 0]
        test = test.drop('train', axis=1)
        X = train.values[:,:4]
        targets = [[1,0,0],[0,1,0],[0,0,1]]
        y = np.array([targets[int(x)] for x in train.values[:,-1]])

        num_inputs = len(X[0])
        hidden_layer_neurons = 5
        np.random.seed(4)
        w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1

        num_outputs = len(y[0])
        w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1

        # TRAINING
        learning_rate = 0.2 
        error = []
        for epoch in range(1000):
            l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
            l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
            er = (abs(y - l2)).mean()
            error.append(er)
            
            # BACKPROPAGATION / learning!
            # find contribution of error on each weight on the second layer
            l2_delta = (y - l2)*(l2 * (1-l2))
            w2 += l1.T.dot(l2_delta) * learning_rate
            
            l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
            w1 += X.T.dot(l1_delta) * learning_rate
        
        #TEST
        X = test.values[:,:4]
        y = np.array([targets[int(x)] for x in test.values[:,-1]])

        l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
        l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

        np.round(l2,3)

        # print(w1)
        # print(w2)

        y_pred = np.argmax(l2, axis=1) 
        res = y_pred == np.argmax(y, axis=1)
        correct = np.sum(res)/len(res)

        test_df = test
        test_df[['Species']] = test[['Species']].replace(range(len(unique)), unique)

        test_df['Prediction'] = y_pred
        test_df['Prediction'] = test_df['Prediction'].replace(range(len(unique)), unique)

        acc = correct
        
        ac_label = tk.Label(window2,width=100, text='Accuracy : '+ str(acc),font=my_font1)
        ac_label.grid(row=2,column=0)

        tv2 = ttk.Treeview(window2,height=10)
        tv2.grid(column=0,row=5,padx=5,pady=10,columnspan=10)
        tv2["column"] = list(test_df.columns)
        tv2["show"] = "headings"
        for column in tv2["columns"]:
                tv2.heading(column, text=column)

        df_rows = test_df.to_numpy().tolist()
        for row in df_rows:
            tv2.insert("", "end", values=row)

        cfm = confusion_matrix(test_df[['Species']], test_df[['Prediction']])
        cm = tk.Text(window2)
        cm.delete("1.0","end")
        cm.grid(row=6,column=0)
        cm.insert(tk.END,'Confusion Matrix: \n' + str(cfm)+'\n')
        cm.insert(tk.END,'\na) Recognition rate: '+str(acc*100))
        cm.insert(tk.END,'\nb) Misclassification rate'+str((1-acc)*100)+"\n\n")
        cm.insert(tk.END,classification_report(test_df[['Species']], test_df[['Prediction']]))

        plt.title('Error Graph')
        plt.plot(error)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()
        
    window2.mainloop()

#######################################################################################################

def upload_file():
    f_types = [('CSV files',"*.csv"),('All',"*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    l1.config(text=file) # display the path 
    df=pd.read_csv(file) # create DataFrame

    tv1 = ttk.Treeview(my_w,height=10)
    tv1.grid(column=0,row=8,padx=5,pady=10,columnspan=10)
    tv1["column"] = list(df.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)
    
    df_rows = df.to_numpy().tolist()
    for row in df_rows:
        tv1.insert("", "end", values=row)

    label = tk.Label(my_w,width=40, text='Select Classifier',font=my_font1)
    label.grid(column=0,row=10,padx=10,pady=10)
    options = ['Logistic Regression classifier','Naïve Bayesian Classifier','k-NN classifier','Three layer Artificial Neural Network (ANN) classifier']
    selected = StringVar(my_w)
    selected.set('Logistic Regression classifier')
    drop_down = tk.OptionMenu(my_w,selected,*options)
    drop_down.grid(column=1,row=10,padx=10,pady=10)


    label2 = tk.Label(my_w,width=40, text='Select train dataset %',font=my_font1)
    label2.grid(column=3,row=10,padx=10,pady=10)
    options2 = [20,30,50,80]
    selected2 = StringVar(my_w)
    selected2.set(str(20))
    drop_down2 = tk.OptionMenu(my_w,selected2,*options2)
    drop_down2.grid(column=4,row=10,padx=10,pady=10)

    q2Button = tk.Button(my_w, text='Compute',width= 40,command = lambda:calculate(df,selected.get(),int(selected2.get())))
    q2Button.grid(row=11,column=0,pady=50)

    label3 = tk.Label(my_w,width=40, text='Select attribute measure : ',font=my_font1)
    label3.grid(column=0,row=12,padx=10,pady=50)
    options3 = ['Information Gain','Gini Ratio','Gini Index']
    selected3 = StringVar(my_w)
    selected3.set('Information Gain')
    drop_down3 = tk.OptionMenu(my_w,selected3,*options3)
    drop_down3.grid(column=1,row=12,padx=10,pady=50)

    q3Button = tk.Button(my_w, text='Implement Decision Tree', 
        width=40,command = lambda:implementDT(df,selected3.get()))
    q3Button.grid(row=12,column=2,pady=50)

my_w.mainloop()  # Keep the window open