#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('Internship original project data set.csv')
data.head()


# In[3]:


#Rename columns
data.columns = ['Timestamp','Gender', 'Age', 'Course', 'Year', 'CGPA', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
data.head(1)


# In[4]:


def Clean(Text):
    Text = Text[-1]
    Text = int(Text)
    return Text
data["Year"] = data["Year"].apply(Clean)
data.head()


# In[5]:


def remove_space(string):
    string = string.strip()
    return string
data["CGPA"] = data["CGPA"].apply(remove_space)
data.head()


# In[6]:


data['CGPA'].unique()


# In[7]:


d={'3.00 - 3.49':0,'3.50 - 4.00':1,'2.50 - 2.99':2,'2.00 - 2.49':3,'0 - 1.99':4}
data['CGPA']=data['CGPA'].map(d)
data.head()


# In[8]:


# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
data['Timestamp']= label_encoder.fit_transform(data['Timestamp'])
data['Gender']= label_encoder.fit_transform(data['Gender'])

data['Course']= label_encoder.fit_transform(data['Course'])

data['Marital_Status']= label_encoder.fit_transform(data['Marital_Status'])

data.head()


# In[9]:


data.isnull().sum()


# In[10]:


data = data.dropna()


# In[11]:


data.isnull().sum()


# In[12]:


x = data.drop(["Depression","Anxiety","Panic_Attack","Treatment"],axis=1)
y = data["Depression"]
y.head()


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

#Check shape

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[14]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dtree=DecisionTreeClassifier(max_depth=3)
dtree= dtree.fit(x_train,y_train)
tree.plot_tree(dtree)


# In[15]:


x = data.drop(["Anxiety","Depression","Panic_Attack","Treatment"],axis=1)
y = data["Anxiety"]
y.head()


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

#Check shape

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dtree1=DecisionTreeClassifier(max_depth=3)
dtree1= dtree1.fit(x_train,y_train)
tree.plot_tree(dtree1)


# In[18]:


x = data.drop(["Panic_Attack","Anxiety","Depression","Treatment"],axis=1)
y = data["Panic_Attack"]
y.head()


# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

#Check shape

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[20]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dtree2=DecisionTreeClassifier(max_depth=3)
dtree2= dtree2.fit(x_train,y_train)
tree.plot_tree(dtree2)


# In[21]:


x = data.drop(["Treatment","Anxiety","Depression","Panic_Attack"],axis=1)
y = data["Treatment"]
y.head()


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

#Check shape

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dtree3=DecisionTreeClassifier(max_depth=3)
dtree3= dtree3.fit(x_train,y_train)
tree.plot_tree(dtree3)


# In[ ]:


# tkinter GUI
import tkinter as tk
root= tk.Tk()

#Make a Canvas (i.e, a screen for your project
canvas1 = tk.Canvas(root, width = 980, height = 400,bg='light yellow')
canvas1.pack()

#To see the GUI screen
root.mainloop()


# In[28]:


#tkinter GUI
import tkinter as tk
root=tk.Tk()


#Make a canvas(a screen for your project)
canvas1 = tk.Canvas(root, width = 980, height = 400,bg='light yellow')
canvas1.pack()


# outlook label and input box


label1 = tk.Label(root, text=' Mental Health Prediction App')
canvas1.create_window(240, 20, window=label1)

label2 = tk.Label(root, text=' Timestamp:')
canvas1.create_window(50, 50, window=label2)

entry1 = tk.Entry (root,bg='light blue') # create 1st entry box
canvas1.create_window(220, 50, window=entry1)

label3 = tk.Label(root, text=' Gender:')
canvas1.create_window(50, 50, window=label2)

entry2 = tk.Entry (root,bg='light blue') # create 1st entry box
canvas1.create_window(220, 50, window=entry1)

label4 = tk.Label(root, text=' Age: ')
canvas1.create_window(50, 75, window=label3)

entry3 = tk.Entry (root,bg='light blue') # create 2nd entry box
canvas1.create_window(220, 75, window=entry2)

# outlook label and input box
label5 = tk.Label(root, text=' Course: ')
canvas1.create_window(50, 100, window=label4)

entry4 = tk.Entry (root,bg='light blue') # create 3rd entry box
canvas1.create_window(220, 100, window=entry3)

# outlook label and input box
label6 = tk.Label(root, text=' Year: ')
canvas1.create_window(50, 125, window=label5)

entry5 = tk.Entry (root,bg='light blue') # create 4th entry box
canvas1.create_window(220, 125, window=entry4)


# outlook label and input box
label7 = tk.Label(root, text=' CGPA: ')
canvas1.create_window(50, 150, window=label6)

entry6 = tk.Entry (root,bg='light blue') # create 5th entry box
canvas1.create_window(220, 150, window=entry5)

# outlook label and input box
label8 = tk.Label(root, text=' Marital_Status: ')
canvas1.create_window(50, 175, window=label7)

entry7 = tk.Entry (root,bg='light blue') # create 6th entry box
canvas1.create_window(220, 175, window=entry6)



#To see the GUI screen
root.mainloop()


# In[29]:


#tkinter GUI
import tkinter as tk
root=tk.Tk()

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

#Make a canvas(a screen for your project)
canvas1 = tk.Canvas(root, width = 980, height = 400,bg='light yellow')
canvas1.pack()

# outlook label and input box

label1 = tk.Label(root, text=' Mental Health Prediction App')
canvas1.create_window(240, 20, window=label1)

label2 = tk.Label(root, text=' Timestamp:')
canvas1.create_window(50, 50, window=label2)

entry1 = tk.Entry (root,bg='light blue') # create 1st entry box
canvas1.create_window(220, 50, window=entry1)

label3 = tk.Label(root, text=' Gender: ')
canvas1.create_window(50, 70, window=label3)

entry2 = tk.Entry (root,bg='light blue') # create 2nd entry box
canvas1.create_window(220, 70, window=entry2)

# outlook label and input box
label4 = tk.Label(root, text=' Age: ')
canvas1.create_window(50, 90, window=label4)

entry3 = tk.Entry (root,bg='light blue') # create 3rd entry box
canvas1.create_window(220, 90, window=entry3)

# outlook label and input box
label5 = tk.Label(root, text=' Course: ')
canvas1.create_window(50, 110, window=label5)

entry4 = tk.Entry (root,bg='light blue') # create 4th entry box
canvas1.create_window(220, 110, window=entry4)


# outlook label and input box
label6 = tk.Label(root, text=' Year: ')
canvas1.create_window(50, 130, window=label6)

entry5 = tk.Entry (root,bg='light blue') # create 5th entry box
canvas1.create_window(220, 130, window=entry5)

# outlook label and input box
label7 = tk.Label(root, text=' CGPA: ')
canvas1.create_window(50, 150, window=label7)

entry6 = tk.Entry (root,bg='light blue') # create 6th entry box
canvas1.create_window(220, 150, window=entry6)


# outlook label and input box
label8 = tk.Label(root, text=' Marital_Status: ')
canvas1.create_window(50, 170, window=label8)

entry7 = tk.Entry (root,bg='light blue') # create 7th entry box
canvas1.create_window(220, 170, window=entry7)

#def values():
# label_Prediction = tk.Label(root, text= 'Hello', bg='light blue')
# canvas1.create_window(190, 170, window=label_Prediction)

def values1():
    
    #global Close #our 1st input variable
    Timestamp = int(entry1.get())
    #global Close #our 2nd input variable
    Gender = int(entry2.get())
    #global Open #our 3rd input variable
    Age = float(entry3.get())
    #global Open #our 4th input variable
    Course = int(entry4.get())
    #global Open #our 5th input variable
    Year = int(entry5.get())
    #global Open #our 6th input variable
    CGPA = int(entry6.get())
    #global Open #our 7th input variable
    Marital_Status = int(entry7.get())
    Prediction_result1 = ('Do you have depression?: ', dtree.predict([[Timestamp,Gender,Age,Course,Year,CGPA,Marital_Status]]))
    label_Prediction1 = tk.Label(root, text= Prediction_result1, bg='pink')
    canvas1.create_window(465, 265, window=label_Prediction1)
    
def values2():
    
    #global Close #our 1st input variable
    Timestamp = int(entry1.get())
    #global Close #our 2nd input variable
    Gender = int(entry2.get())
    #global Open #our 3rd input variable
    Age = float(entry3.get())
    #global Open #our 4th input variable
    Course = int(entry4.get())
    #global Open #our 5th input variable
    Year = int(entry5.get())
    #global Open #our 6th input variable
    CGPA = int(entry6.get())
    #global Open #our 7th input variable
    Marital_Status = int(entry7.get())
    Prediction_result2 = ('Do you have anxiety?: ', dtree1.predict([[Timestamp,Gender,Age,Course,Year,CGPA,Marital_Status]]))
    label_Prediction2 = tk.Label(root, text= Prediction_result2, bg='pink')
    canvas1.create_window(465, 300, window=label_Prediction2)
    
def values3():
    
    #global Close #our 1st input variable
    Timestamp = int(entry1.get())
    #global Close #our 2nd input variable
    Gender = int(entry2.get())
    #global Open #our 3rd input variable
    Age = float(entry3.get())
    #global Open #our 4th input variable
    Course = int(entry4.get())
    #global Open #our 5th input variable
    Year = int(entry5.get())
    #global Open #our 6th input variable
    CGPA = int(entry6.get())
    #global Open #our 7th input variable
    Marital_Status = int(entry7.get())
    Prediction_result3 = ('Do you have panic attack?: ', dtree2.predict([[Timestamp,Gender,Age,Course,Year,CGPA,Marital_Status]]))
    label_Prediction3 = tk.Label(root, text= Prediction_result3, bg='pink')
    canvas1.create_window(465, 335, window=label_Prediction3)
    
def values4():
    
    #global Close #our 1st input variable
    Timestamp = int(entry1.get())
    #global Close #our 2nd input variable
    Gender = int(entry2.get())
    #global Open #our 3rd input variable
    Age = float(entry3.get())
    #global Open #our 4th input variable
    Course = int(entry4.get())
    #global Open #our 5th input variable
    Year = int(entry5.get())
    #global Open #our 6th input variable
    CGPA = int(entry6.get())
    #global Open #our 7th input variable
    Marital_Status = int(entry7.get())
    Prediction_result4 = ('Do you need specialist treatment?: ', dtree3.predict([[Timestamp,Gender,Age,Course,Year,CGPA,Marital_Status]]))
    label_Prediction4 = tk.Label(root, text= Prediction_result4, bg='pink')
    canvas1.create_window(465, 370, window=label_Prediction4)
    
def message_bx():
    top = tk.Toplevel()

    top.geometry("700x600")

    top["bg"] = "#A4DE02"
    
    text1 ="""
    
    COMMON CONCLUSIONS
    
    *******************************************************************************************
    1) We can generalize the results and conclude that female students are more likely to suffer 
    from depression compared to males
    
    2) The number of female students having anxiety is more but the percentage of female students 
    suffering from anxiety is lower than that of male students suffering from anxiety.
    
    3)The year-anxiety graph tells us that year 4 has the lowest percentage of students suffering 
    from anxiety(25%) whereas year 2 has the highest percentage of students having anxiety(about 38.5%).
    
    4)The students having the lowest CGPA are less likely to have panic attacks.
    
    """
    current_label1 = tk.Label(top, text = text1, justify= 'center', width=390,background="orange", foreground='blue')
    
    current_label1.pack()
    top.mainloop()
    

button5= tk.Button (root, text='Display conclusions',command=message_bx, bg='pink') # button to call the 'message_bx' command above
canvas1.create_window(190, 230, window=button5)
button1 = tk.Button (root, text='Do you have depression?:',command=values1, bg='pink') # button to call the 'values1' command above
canvas1.create_window(190, 265, window=button1)
button2 = tk.Button (root, text='Do you have anxiety?:',command=values2, bg='pink') # button to call the 'values2' command above
canvas1.create_window(190, 300, window=button2)
button3 = tk.Button (root, text='Do you have panic attack?:',command=values3, bg='pink') # button to call the 'values3' command above
canvas1.create_window(190, 335, window=button3)
button4 = tk.Button (root, text='Do you need specialist treatment?:',command=values4, bg='pink') # button to call the 'values4' command above
canvas1.create_window(190, 370, window=button4)

sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
filter1 =  data["Anxiety"] == 'Yes'
data1 = data[filter1].groupby("Year")['Anxiety'].count() / data.groupby("Year")['Anxiety'].count() * 100
data2 = 100 - data1
combine_data = pd.concat([data1, data2],axis=1)
combine_data.columns = ["Yes", "No"]
combine_data.plot(kind='bar', figsize=(10,6), xlabel="Your current year of Study", ylabel="Have Anxiety", title="Students having anxiety based on study years",ax=axes[0, 0])

filter2 = data["Depression"] == 'Yes'
dp = data[filter2].groupby("Year")["Depression"].count() / data.groupby("Year")["Depression"].count() * 100
dp1 = 100 - dp
combine_dp = pd.concat([dp, dp1], axis=1)
combine_dp.columns = ["Yes", "No"]
combine_data.plot(kind='bar', figsize=(10,6), xlabel="Your current year of Study", ylabel="Have Depression", title="Students having depression based on study years",ax=axes[0, 1])

#Students who have panic attack based on CGPA
filter3 =  data["Panic_Attack"] == 'Yes'
data3 = data[filter3].groupby("CGPA")['Panic_Attack'].count() / data.groupby("CGPA")['Panic_Attack'].count() * 100
d4 = 100 - data3
combine_data = pd.concat([data3, d4],axis=1)
combine_data.columns = ["Yes", "No"]
combine_data.plot(kind='bar', figsize=(10,6), xlabel="CGPA", ylabel="Have frequent panic attacks", title="Students having panic attack based on their CGPA and performance in college",ax=axes[1, 0])

#Students who are under specialised treatment based on depression

filter4 =  data["Treatment"] == 'Yes'
data4 = data[filter4].groupby("Depression")['Treatment'].count() / data.groupby("Depression")['Treatment'].count() * 100
d5 = 100 - data4
combine_data = pd.concat([data4, d5],axis=1)
combine_data.columns = ["Yes", "No"]
combine_dp.plot(kind='bar',xlabel="Depression", ylabel="Percentage of students who are undergoing treatment", title="Average number of students under treatment based on depression",figsize=(10,6),ax=axes[1, 1])

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

#To see the GUI screen
root.mainloop()


# In[ ]:




