import pandas as pd
from tkinter import *
import tkinter as tk
from tkinter import ttk
import math
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

my_w = tk.Tk()
my_w.geometry("800x500")  # Size of the window 
my_w.title('DATA VISUALIZE')

my_font1=('times', 12, 'bold')
l1 = tk.Label(my_w,width=40, text='Select & Read File',font=my_font1)  
l1.grid(row=1)
b1 = tk.Button(my_w, text='Browse File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2) 
# t1=tk.Text(my_w,width=150,height=15)
# t1.grid(row=3,column=1,padx=5)

def compute(query,data,attr1,attr2,target):
                    if query.get() == "Chi-Square Test":
                        window2 = Tk()
                        window2.title(query.get())
                        window2.geometry("500x500")

                        cols = []
                        for i in data.columns:
                            cols.append(i)

                        attribute1 = attr1.get()
                        attribute2 = attr2.get()
                        category = target.get()
                        arrClass = data[category].unique()
                        g = data.groupby(category)
                        f = {
                            attribute1: 'sum',
                            attribute2: 'sum'
                        }
                        v1 = g.agg(f)
                        print(v1)
                        v = v1.transpose()
                        print(v)
                        
                        tv1 = ttk.Treeview(window2,height=3)
                        tv1.grid(column=1,row=8,padx=5,pady=8)
                        tv1["column"] = list(v.columns)
                        tv1["show"] = "headings"
                        for column in tv1["columns"]:
                            tv1.heading(column, text=column)

                        df_rows = v.to_numpy().tolist()
                        for row in df_rows:
                                tv1.insert("", "end", values=row)

                        total = v1[attribute1].sum()+v1[attribute2].sum()
                        chiSquare = 0.0
                        for i in arrClass:
                                chiSquare += (v.loc[attribute1][i]-(((v[i].sum())*(v1[attribute1].sum()))/total))*(v.loc[attribute1][i]-(((v[i].sum())*(v1[attribute1].sum()))/total))/(((v[i].sum())*(v1[attribute1].sum()))/total)
                                chiSquare += (v.loc[attribute2][i]-(((v[i].sum())*(v1[attribute2].sum()))/total))*(v.loc[attribute2][i]-(((v[i].sum())*(v1[attribute2].sum()))/total))/(((v[i].sum())*(v1[attribute2].sum()))/total)
                            
                        degreeOfFreedom = (len(v)-1)*(len(v1)-1)
                        Label(window2,text="Chi-square value is "+str(chiSquare), justify='center',height=2,fg="green").grid(column=1,row=9,padx=5,pady=8) 
                        Label(window2,text="Degree of Freedom is "+str(degreeOfFreedom), justify='center',height=2,fg="green").grid(column=1,row=10,padx=5,pady=8) 
                        res = ""
                        if chiSquare > degreeOfFreedom:
                                res = "Attributes " + attribute1 + ' and ' + attribute2 + " are strongly correlated."
                        else:
                                res = "Attributes " + attribute1 + ' and ' + attribute2 + " are not correlated."
                        Label(window2,text=res, justify='center',height=2,fg="green").grid(column=1,row=11,padx=5,pady=8)
                        window2.mainloop()

                    elif query.get() == "Correlation(Pearson) Coefficient":
                        window2 = Tk()
                        window2.title(query.get())
                        window2.geometry("500x500")
                        cols = []
                        for i in data.columns:
                            cols.append(i)
                        
                        attribute1 = attr1.get()
                        attribute2 = attr2.get()
                            
                        sum = 0
                        for i in range(len(data)):
                                sum += data.loc[i, attribute1]
                        avg1 = sum/len(data)
                        sum = 0
                        for i in range(len(data)):
                                sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute1]-avg1)
                        var1 = sum/(len(data))
                        sd1 = math.sqrt(var1)
                            
                        sum = 0
                        for i in range(len(data)):
                                sum += data.loc[i, attribute2]
                        avg2 = sum/len(data)
                        sum = 0
                        for i in range(len(data)):
                                sum += (data.loc[i, attribute2]-avg2)*(data.loc[i, attribute2]-avg2)
                        var2 = sum/(len(data))
                        sd2 = math.sqrt(var2)
                        
                        sum = 0
                        for i in range(len(data)):
                                sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute2]-avg2)
                        covariance = sum/len(data)
                        pearsonCoeff = covariance/(sd1*sd2)    
                        Label(window2,text="Covariance value is "+str(covariance), justify='center',height=2,fg="green").grid(column=1,row=8,padx=5,pady=8) 
                        Label(window2,text="Correlation coefficient(Pearson coefficient) is "+str(pearsonCoeff), justify='center',height=2,fg="green").grid(column=1,row=9,padx=5,pady=8) 
                        res = ""
                        if pearsonCoeff > 0:
                                res = "Attributes " + attribute1 + ' and ' + attribute2 + " are positively correlated."
                        elif pearsonCoeff < 0:
                                res = "Attributes " + attribute1 + ' and ' + attribute2 + " are negatively correlated."
                        elif pearsonCoeff == 0:
                                res = "Attributes " + attribute1 + ' and ' + attribute2 + " are independant."
                        Label(window2,text=res, justify='center',height=2,fg="green").grid(column=1,row=11,padx=5,pady=8)
                        window2.mainloop()

                    elif query.get() == "Normalization Techniques":
                        window2 = Tk()
                        window2.title(query.get())
                        window2.geometry("500x500")
                        cols = []
                        for i in data.columns:
                            cols.append(i)
                        
                        normalizationOperations = ["Min-Max normalization","Z-Score normalization","Normalization by decimal scaling"]
                        clickedOperation = StringVar(window2)
                        clickedOperation.set("Select Normalization Operation")
                        dropOperations = OptionMenu(window2, clickedOperation, *normalizationOperations)
                        dropOperations.grid(column=4,row=5)
                        Button(window2,text="Compute",command= lambda:computeOperation()).grid(column=2,row=7,padx=20,pady=30) 
                        
                        def computeOperation():
                            attribute1 = attr1.get()
                            attribute2 = attr2.get() 
                            operation = clickedOperation.get()
                            if operation == "Min-Max normalization":
                                n = len(data)
                                arr1 = []
                                for i in range(len(data)):
                                    arr1.append(data.loc[i, attribute1])
                                arr1.sort()
                                min1 = arr1[0]
                                max1 = arr1[n-1]
                                
                                arr2 = []
                                for i in range(len(data)):
                                    arr2.append(data.loc[i, attribute2])
                                arr2.sort()
                                min2 = arr2[0]
                                max2 = arr2[n-1]
                                
                                for i in range(len(data)):
                                    data.loc[i, attribute1] = ((data.loc[i, attribute1]-min1)/(max1-min1))
                                
                                for i in range(len(data)):
                                    data.loc[i, attribute2] = ((data.loc[i, attribute2]-min2)/(max2-min2))
                            elif operation == "Z-Score normalization":
                                sum = 0
                                for i in range(len(data)):
                                    sum += data.loc[i, attribute1]
                                avg1 = sum/len(data)
                                sum = 0
                                for i in range(len(data)):
                                    sum += (data.loc[i, attribute1]-avg1)*(data.loc[i, attribute1]-avg1)
                                var1 = sum/(len(data))
                                sd1 = math.sqrt(var1)
                                
                                sum = 0
                                for i in range(len(data)):
                                    sum += data.loc[i, attribute2]
                                avg2 = sum/len(data)
                                sum = 0
                                for i in range(len(data)):
                                    sum += (data.loc[i, attribute2]-avg2)*(data.loc[i, attribute2]-avg2)
                                var2 = sum/(len(data))
                                sd2 = math.sqrt(var2)
                                
                                for i in range(len(data)):
                                    data.loc[i, attribute1] = ((data.loc[i, attribute1]-avg1)/sd1)
                                
                                for i in range(len(data)):
                                    data.loc[i, attribute2] = ((data.loc[i, attribute2]-avg2)/sd2)
                            elif operation == "Normalization by decimal scaling":        
                                j1 = 0
                                j2 = 0
                                n = len(data)
                                arr1 = []
                                for i in range(len(data)):
                                    arr1.append(data.loc[i, attribute1])
                                arr1.sort()
                                max1 = arr1[n-1]
                                
                                arr2 = []
                                for i in range(len(data)):
                                    arr2.append(data.loc[i, attribute2])
                                arr2.sort()
                                max2 = arr2[n-1]
                                
                                while max1 > 1:
                                    max1 /= 10
                                    j1 += 1
                                while max2 > 1:
                                    max2 /= 10
                                    j2 += 1
                                
                                for i in range(len(data)):
                                    data.loc[i, attribute1] = ((data.loc[i, attribute1])/(pow(10,j1)))
                                
                                for i in range(len(data)):
                                    data.loc[i, attribute2] = ((data.loc[i, attribute2])/(pow(10,j2)))
                            
                            Label(window2,text="Normalized Attributes", justify='center',height=2,fg="green").grid(column=1,row=8,padx=5,pady=8)         
                            tv1 = ttk.Treeview(window2,height=15)
                            tv1.grid(column=1,row=9,padx=5,pady=8)
                            tv1["column"] = [attribute1,attribute2]
                            tv1["show"] = "headings"
                            for column in tv1["columns"]:
                                tv1.heading(column, text=column)
                            i = 0
                            while i < len(data):
                                tv1.insert("", "end", iid=i, values=(data.loc[i, attribute1],data.loc[i, attribute2]))
                                i += 1
                            sns.set_style("whitegrid")
                            sns.FacetGrid(data, hue=target.get(), height=4).map(plt.scatter, attribute1, attribute2).add_legend()
                            plt.title("Scatter plot")
                            plt.show(block=True)
                        window2.mainloop()

def calculate(query,df):

    columns = df.columns
    cols = columns[1:]

    
    attribute1 = StringVar()
    attribute2 = StringVar()
    target = StringVar()

    option1 = OptionMenu(my_w,attribute1,*cols).grid(row=4,column=0,pady=10,padx=5)
    option1 = OptionMenu(my_w,attribute2,*cols).grid(row=4,column=1,pady=10,padx=5)
    classes = OptionMenu(my_w,target,*cols).grid(row=4,column=2,pady=10,padx=5)
    q2Button = tk.Button(my_w, text='compute', 
    width=30,command = lambda:compute(query,df,attribute1,attribute2,target))
    q2Button.grid(row=4,column=3,pady=10,padx=10)


def upload_file():
    f_types = [('CSV files',"*.csv"),('All',"*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    l1.config(text=file) # display the path 
    df=pd.read_csv(file) # create DataFrame
    # str1=df
    # t1.delete("1.0","end")
    # t1.insert(tk.END, str1) # add to Text widget

    # Menu bar
    clicked = StringVar()
    clicked.set( "Choose" )

    values = ['Chi-Square Test','Correlation(Pearson) Coefficient','Normalization Techniques']

    option_menu = OptionMenu(my_w,clicked,*values).grid(row=3,column=0,pady=10)

    q2Button = tk.Button(my_w, text='calculate', 
        width=40,command = lambda:calculate(clicked,df))
    q2Button.grid(row=3,column=1,pady=10,padx=10)
    
my_w.mainloop()  # Keep the window open