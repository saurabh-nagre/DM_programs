from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

def browseDataset():
    filename = filedialog.askopenfilename(initialdir="/",title="Select dataset", filetypes=(("CSV files", "*.csv*"), ("all files", "*.*")))
    label_file_explorer.configure(text="File Opened: "+filename)
    newfilename = ''
    for i in filename:
        if i == "/":
            newfilename = newfilename + "/"
        newfilename = newfilename + i
    data = pd.read_csv(filename)
    cols = []
    for i in data.columns:
        cols.append(i)
    clickedAttribute1 = StringVar()
    clickedAttribute1.set("Select Attribute 1")
    clickedAttribute2 = StringVar()
    clickedAttribute2.set("Select Attribute 2")
    clickedClass = StringVar()
    clickedClass.set("Select class")
    plots = ["Quantile-Quantile Plot","Histogram","Scatter Plot","Boxplot"]
    clickedPlot = StringVar()
    clickedPlot.set("Select Plot")
    dropPlots = OptionMenu(window, clickedPlot, *plots)
    dropPlots.grid(column=1,row=6)
    Button(window,text="Select Attributes",command= lambda:selectAttributes()).grid(column=2,row=6)
    
    def computeOperation():
        attribute1 = clickedAttribute1.get()
        attribute2 = clickedAttribute2.get()
        
        operation = clickedPlot.get()
        if operation == "Quantile-Quantile Plot": 
            arr = []
            sum = 0
            for i in range(len(data)):
                arr.append(data.loc[i, attribute1])  
                sum += data.loc[i, attribute1]
            avg = sum/len(arr)
            sum = 0
            for i in range(len(data)):
                sum += (data.loc[i, attribute1]-avg)*(data.loc[i, attribute1]-avg)
            var = sum/(len(data))
            sd = math.sqrt(var)
            z = (arr-avg)/sd
            stats.probplot(z, dist="norm", plot=plt)
            plt.title("Normal Q-Q plot")
            plt.show()
            
        elif operation == "Histogram": 
            sns.set_style("whitegrid")
            sns.FacetGrid(data, hue=clickedClass.get(), height=5).map(sns.histplot, attribute1).add_legend()
            plt.title("Histogram")
            plt.show(block=True)
        elif operation == "Scatter Plot":
            sns.set_style("whitegrid")
            sns.FacetGrid(data, hue=clickedClass.get(), height=4).map(plt.scatter, attribute1, attribute2).add_legend()
            plt.title("Scatter plot")
            plt.show(block=True)
        elif operation == "Boxplot":
            sns.set_style("whitegrid")
            sns.boxplot(x=attribute1,y=attribute2,data=data)
            plt.title("Boxplot")
            plt.show(block=True)
        
    def selectAttributes():
        operation = clickedPlot.get()
        if operation == "Quantile-Quantile Plot":
            dropCols = OptionMenu(window, clickedAttribute1, *cols)
            dropCols.grid(column=3,row=8,padx=20,pady=30)  
            Button(window,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)
        
        elif operation == "Histogram":   
            dropCols = OptionMenu(window, clickedAttribute1, *cols)
            dropCols.grid(column=3,row=8,padx=20,pady=30)  
            dropCols = OptionMenu(window, clickedClass, *cols)
            dropCols.grid(column=5,row=8,padx=20,pady=30) 
            Button(window,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)
    
        elif operation == "Scatter Plot":
            dropCols = OptionMenu(window, clickedAttribute1, *cols)
            dropCols.grid(column=2,row=8,padx=20,pady=30)
            dropCols = OptionMenu(window, clickedAttribute2, *cols)
            dropCols.grid(column=3,row=8,padx=20,pady=30)
            dropCols = OptionMenu(window, clickedClass, *cols)
            dropCols.grid(column=5,row=8)
            Button(window,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)

        elif operation == "Boxplot":
            dropCols = OptionMenu(window, clickedAttribute1, *cols)
            dropCols.grid(column=2,row=8,padx=20,pady=30)
            dropCols = OptionMenu(window, clickedAttribute2, *cols)
            dropCols.grid(column=3,row=8,padx=20,pady=30)
            Button(window,text="Compute",command= lambda:computeOperation()).grid(column=4,row=6)
            
            
window = Tk()
window.title("Plots of data")
window.geometry("500x500")
window.config(background="white")
label_file_explorer = Label(window,text="Choose Dataset from File Explorer",justify='center',height=4,fg="blue")
button_explore = Button(window,text="Browse Dataset",command=browseDataset)
button_exit = Button(window,text="Exit",command=exit)
label_file_explorer.grid(column=1,row=1,padx=20,pady=30)
button_explore.grid(column=3,row=1,padx=20,pady=30)
button_exit.grid(column=5,row=1,padx=20,pady=30)
window.mainloop()