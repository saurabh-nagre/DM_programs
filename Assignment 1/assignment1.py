import math
import pandas as pd
import tkinter  as tk 
from tkinter import *
from tkinter import filedialog
my_w = tk.Tk()
my_w.geometry("800x500")  # Size of the window 
my_w.title('DATA VISUALIZE')

my_font1=('times', 12, 'bold')
l1 = tk.Label(my_w,text='Select & Read File',
    width=30,font=my_font1)  
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Browse File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1) 
t1=tk.Text(my_w,width=100,height=10)
t1.grid(row=3,column=1,padx=5)

def calculateMean(arr):
    mean = 0
    for i in arr:
        mean+=i
    return str(mean/len(arr))

def calculateMedian(arr):
    l = len(arr)/2
    if l%2==0:
        med = (arr[l]+arr[l-1])/2
        return str(med)
    else :
        return str(arr[l])
        
def calculateMode(arr):
    maxCount = 0
    list = []
    for ele in arr:
        list.append(ele)
        
    for ele in list:
        maxCount = max(maxCount,list.count(ele))
    
    res = set({})
    for ele in list:
        if list.count(ele)==maxCount:
            res.add(ele)
    return str(res)

def calculateMidRange(arr):
    return str((min(arr)+max(arr))/2)

def calculateVariance(arr):
    mean = float(calculateMean(arr))
    sum = 0
    for ele in arr:
        sum+=math.pow(ele-mean,2)
    return str(sum/(len(arr)-1))

def calculateSD(arr):
    variance = float(calculateVariance(arr))
    return str(math.sqrt(variance))

def displayCentralTendancy(df):
    meanLabel = Label(text="Mean: ",width=20,height=1).place(y=220)
    medianLabel = Label(text="Median: ",width=20,height=1).place(y =320)
    modeLabel  = Label(text='Mode: ',width=20,height=1).place(y =420)
    midrangeLabel = Label(text='Mid Range: ',width=20,height=1).place(y =520)
    varianceLabel =  Label(text='Variance: ',width=20,height=1).place(y =620)
    standard_deviationLabel =  Label(text='Std Deviation: ',width=20,height=1).place(y =720)


    meanText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        meanText.insert(tk.END,attr+": "+calculateMean(df.loc[:,attr])+"\n")
    meanText.place(x= 150,y =220)

    medianText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        df.sort_values(by=[attr])
        medianText.insert(tk.END,attr+": "+calculateMedian(df.loc[:,attr])+"\n")
    medianText.place(x= 150,y =320)

    modeText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        df.sort_values(by=[attr])
        modeText.insert(tk.END,attr+": "+calculateMode(df.loc[:,attr])+"\n")
    modeText.place(x= 150,y =420)

    midrangeText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        midrangeText.insert(tk.END,attr+": "+calculateMidRange(df.loc[:,attr])+"\n")
    midrangeText.place(x= 150,y =520)

    varianceText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        varianceText.insert(tk.END,attr+": "+calculateVariance(df.loc[:,attr])+"\n")
    varianceText.place(x= 150,y =620)

    sdText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        sdText.insert(tk.END,attr+": "+calculateSD(df.loc[:,attr])+"\n")
    sdText.place(x= 150,y =720)

def calculateRange(arr):
    return str(max(arr)-min(arr))

def calculateQuartiles(arr):
    l = len(arr)//4
    first = arr[l]
    second = arr[l*2]
    third = arr[3*l]
    list = [first,second,third]
    return str(list)

def calculateIQR(arr):
    l = len(arr)
    first = arr[l//4]
    third = arr[3*(l//4)]
    return str(third-first)
 
def calculate5NS(arr):
    l = len(arr)
    first = arr[l//4]
    second = arr[l//2]
    third = arr[3*(l//4)]
    list = [min(arr),first,second,third,max(arr)]
    return str(list)


def displayDispersion(df):
    
    rangeLabel = Label(text="Range: ",width=20,height=1).place(x = 560,y=220)
    quartilesLabel = Label(text="Quartiles: ",width=20,height=1).place(x = 560,y =320)
    interquartile_rangeLabel  = Label(text='interquartile range: ',width=20,height=1).place(x = 560,y =420)
    five_number_summaryLabel = Label(text='five-number summary: ',width=20,height=1).place(x = 560,y =520)


    rangeText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        rangeText.insert(tk.END,attr+": "+calculateRange(df.loc[:,attr])+"\n")
    rangeText.place(x = 700,y =220)

    quartilesText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        df.sort_values(by=[attr])
        quartilesText.insert(tk.END,attr+": "+calculateQuartiles(df.loc[:,attr])+"\n")
    quartilesText.place(x = 700,y =320)

    interQRText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        df.sort_values(by=[attr])
        interQRText.insert(tk.END,attr+": "+calculateIQR(df.loc[:,attr])+"\n")
    interQRText.place(x = 700,y =420)

    fiveNSText = tk.Text(my_w,width=50,height=5,pady=5)
    for attr in df.columns.to_list()[1:-1]:
        df.sort_values(by=[attr])
        fiveNSText.insert(tk.END,attr+": "+calculate5NS(df.loc[:,attr])+"\n")
    fiveNSText.place(x = 700,y =520)



def upload_file():
    f_types = [('CSV files',"*.csv"),('All',"*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    l1.config(text=file) # display the path 
    df=pd.read_csv(file) # create DataFrame
    str1=df
    #print(str1)
    t1.delete("1.0","end")
    t1.insert(tk.END, str1) # add to Text widget

    # q2Button = tk.Button(my_w, text='Calculate Central Tendancy', 
    #     width=40,command = lambda:)
    # q2Button.grid(row=1,column=3,pady=5)

    displayCentralTendancy(df)
    displayDispersion(df)
    # q3Button = tk.Button(my_w, text='Calculate dispersion of data', 
    #     width=40,command = lambda:)
    # q3Button.grid(row=2,column=3,pady=5)

my_w.mainloop()  # Keep the window open
