# IMPORT tkinter library
from tkinter import *
from tkinter import ttk
# Import image library
from PIL import ImageTk, Image
import pandas as pd
# importing the preprocessing packages
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# importing the sampling 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
# for a html eda page
from ydata_profiling import ProfileReport
import webbrowser
# import classification metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# importing spliting package
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # for naive bayes
from sklearn.neighbors import KNeighborsClassifier # for KNN
from sklearn.svm import SVC # for SVM


WIDTH = 800
HEIGHT = 500

store = []

# centre the window
def center_window(window):
    # Update the window to ensure its dimensions are calculated correctly
    window.update_idletasks()

    # Get the window's width and height
    window_width = window.winfo_width()
    window_height = window.winfo_height()

    # Get the screen's width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the x and y coordinates for centering
    # This places the top-left corner of the window at the calculated position
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    # Set the window's geometry using the calculated coordinates
    # The format is "widthxheight+x+y"
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# background image
def background_img(wind):
    #  resize the image
    img = Image.open('8622.jpg')
    resized_bg = img.resize((WIDTH, HEIGHT))
    back_img = ImageTk.PhotoImage(resized_bg)
    back_label = Label(wind, image = back_img)
    back_label.image = back_img
    back_label.place(x=0, y=0, relwidth=1, relheight=1)
    # lower the background so other widgets appear above
    back_label.lower()

def extract(map, var):
    content = var.get()
    store.append((map, content))
    print('Done', store)

# Naive Bayes
def NaiveBayes(X_train, X_test, y_train, y_test):
    # Initialize Gaussian Naive Bayes
    model = GaussianNB()
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    
    return classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)

# KNN 
def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)

# SVM
def SVM(X_train, X_test, y_train, y_test):
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)

# EDA process
def EDA(df, info_window):
    global store
    temp = list(set(store)) # it is used the remove all the duplicate elements
    store = []
    print(temp)
    column = df.columns
    #print(df.columns, df.columns[0])
    label = []
    data_type = []
    im_algo = []
    encoding = []
    drop_col = []
    scaling = []
    for i in temp:
        print(i)
        # determine the label
        if i[0] == 1:
            label.append(i[1])
        # determine the type of data 
        if i[0] == 2:
            data_type.append(i[1])
        # determine the type of algo preferred to solve imbalance data problem
        if i[0] == 3:
            im_algo.append(i[1])
        # attributes need encoding
        if i[0] == 4:
            encoding.append(i[1])
        # columns/attributes to delete
        if i[0] == 5:
            drop_col.append(i[1])
        # for scaling
        if i[0] == 6:
            scaling.append(i[1])
    '''print(label)
    print(data_type)
    print(im_algo)
    print(encoding)'''
    
    y = df.iloc[:, [list(column).index(label[0])]]
    
    if 'No need' not in drop_col:
        df.drop(columns = drop_col, axis=1, inplace=True)
    X = df.drop([list(column).index(label[0])])
    # one hot encoding
    if 'No need' not in encoding:
        df = pd.get_dummies(df, columns=encoding)

    # scaling to a standard value
    print(df.dtypes)
    if scaling[0] == 'StandardScaler':
        scale = StandardScaler()
        X = scale.fit_transform(X) 
        df = pd.DataFrame(X, columns=df.columns)
    if scaling[0] == 'MinMaxScaler':
        scale = MinMaxScaler()
        X = scale.fit_transform(X) 
        df = pd.DataFrame(X, columns=df.columns)
 
    # solve the imbalance data problem

    print(df.shape, y[label[0]].value_counts())

    df[y.columns] = y

    #info_win(df, info_window)
    ml_window(df, info_window, data_type, label)
  

def ml_window(df, old_win, cat_or_num, label):
    # here before destroying the previous window you need to extrct the content of store list and then empty it
    global store 
    temp = store
    store = []
    #print(temp, store)
    old_win.destroy()
    ml_window = Tk()
    ml_window.title('CHOOSE THE ML ALGO')
    ml_window.geometry('600x230')
    ml_window.resizable(False, False)  # Lock the size of the window
    background_img(ml_window)
    # text area
    # Create a Frame to hold Text and Scrollbars
    back_0 = "#5B37EC"
    frame = Frame(ml_window, width=400, height=400, bg=back_0)
    frame.pack(pady=(15, 20))

    # Configure the scrollbars
    #scrollbar_y.config(command=text_area.yview)
    #scrollbar_x.config(command=text_area.xview)

    # now we shall split the data into training and testing
    print(label)
    column = df.columns
    y = df[label[0]]
    df.drop(columns = label, axis=1, inplace=True)
    x = df

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def open_web():
        # EDA full html report
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile.to_file("eda_report.html")
        webbrowser.open("eda_report.html")

    # Different ml algorithms 

    #  Categorical datas with class
    if cat_or_num[0] == 'Categorical':    
        accuracy = []
        precision = []
        f1 = []
        recall=[]
        def status(class_metrics, acc):
            precision.append((class_metrics['0']['precision'], class_metrics['1']['precision']))
            recall.append((class_metrics['0']['recall'], class_metrics['1']['recall']))
            f1.append((class_metrics['0']['f1-score'], class_metrics['1']['f1-score']))
            accuracy.append(acc)
        w = 12
        back = "#A9B4FD"
        Label(frame, text='ALGORITHM', width = w, bg=back_0, fg="white", font= ('Helvetica 10 underline')).grid(row=0, column=0)
        Label(frame, text='0 or 1', width = w, bg=back).grid(row=0, column=1)
        Label(frame, text='PRECISION', width = w, bg=back).grid(row=0, column=2)
        Label(frame, text='RECALL', width = w, bg=back).grid(row=0, column=3)
        Label(frame, text='F1-SCORE', width = w, bg=back).grid(row=0, column=4)
        Label(frame, text='ACCURACY(%)', width = w, bg=back).grid(row=0, column=5)
        # Naive Bayes
        class_metrics, conf_matrix, acc = NaiveBayes(X_train, X_test, y_train, y_test)
        status(class_metrics, acc)
        class_metrics, conf_matrix, acc = KNN(X_train, X_test, y_train, y_test)
        status(class_metrics, acc)
        class_metrics, conf_matrix, acc = SVM(X_train, X_test, y_train, y_test)
        status(class_metrics, acc)
        algo = ["NaiveBaye's", "KNN", "SVM"]
        c = 0
        r = 1
        for i in range(1,6,2):
            Label(frame, text=algo[c], width=w, bg=back_0, fg="white", font=('Aerial', 9,"bold")).grid(row=i, column=0, rowspan=2)
            Label(frame, text=f"{accuracy[c]*100:.3f}", width=w, bg=back_0, fg="white").grid(row=i, column=5, rowspan=2)
            print(i)
            flag = "0"
            for j in range(0,2,1):
                if j == 0:
                    back_1 = "#FDFDFF"
                else:
                    back_1 = "#68686D"
                Label(frame, text=flag, bg=back_1, width=w, relief=RAISED).grid(row=r, column=1)
                Label(frame, text=f"{precision[c][j]:.3f}", bg=back_1, width=w, relief=RAISED).grid(row=r, column=2)
                Label(frame, text=f"{recall[c][j]:.3f}", bg=back_1, width=w, relief=RAISED).grid(row=r, column=3)
                Label(frame, text=f"{f1[c][j]:.3f}", bg=back_1, width=w, relief=RAISED).grid(row=r, column=4)
                flag = "1"
                r+=1
            flag = "0"
            c+=1
        Button(ml_window, text='Click here to get more info about the dataset', command= open_web).pack()
        

    center_window(ml_window)
    ml_window.mainloop()

def info_win(df, root):
    root.destroy()
    #print(path)
    info_window = Tk()
    info_window.title('INFORMATION ABOUT DATASET')
    info_window.geometry(f"{WIDTH}x{HEIGHT}")
    info_window.resizable(False, False)  # Lock the size of the window
    background_img(info_window)

    style = ttk.Style()
    # Choose a theme that supports custom styling (like 'clam')
    style.theme_use("clam")

    # Customize the Treeview heading
    style.configure("Treeview.Heading", 
                background="#4A3A8B",   # background color
                foreground="white",    # text color
                font=('Tahoma', 10, 'bold'))
    
    style.configure("Vertical.TScrollbar",
                     background="#8D83B7",    # scroll bar handle
                darkcolor="#5A4E8C",     # darker edge
                lightcolor="#AFA8D6",    # lighter edge
                troughcolor="#E8E8E8",   # background track
                bordercolor="#CCCCCC",
                arrowcolor="black")
    style.configure("Horizontal.TScrollbar",
                     background="#8D83B7",    # scroll bar handle
                darkcolor="#5A4E8C",     # darker edge
                lightcolor="#AFA8D6",    # lighter edge
                troughcolor="#E8E8E8",   # background track
                bordercolor="#CCCCCC",
                arrowcolor="black")

    # Left and right frames
    bg_right_frame = "#A9B4FD"
    left_frame = Frame(info_window, width=450, height=500, bg='grey')
    right_frame = Frame(info_window, width=340, height=500)

    background_img(right_frame)

    left_frame.grid(row=0, column=0, sticky="nw", padx=(5,0), pady=5)
    right_frame.grid(row=0, column=1, sticky="nw", padx=(0,5), pady=5)

    left_frame.grid_propagate(False)
    right_frame.grid_propagate(False)

    # tree view of csv file
    tree = ttk.Treeview(left_frame)
    tree.pack(fill="both", expand=True)
    # Define columns
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"  # Hide the default first empty column

    # Set headings
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=90, anchor='center')

    # Insert data into treeview
    tree.tag_configure('oddrow', background="#C1BFBF")
    tree.tag_configure('evenrow', background="#ACA4CB")
    count = 0
    for index, row in df.iterrows():
        if count%2 == 0:
            tree.insert("", "end", values=list(row), tags=('oddrow',))
        else:
            tree.insert("", "end", values=list(row), tags=('evenrow',))
        count+=1

    # Add scrollbar
     # Create vertical scrollbar
    vsb = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)

    # Create horizontal scrollbar
    hsb = ttk.Scrollbar(left_frame, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=hsb.set)

    # Use grid to place everything correctly
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    # Make sure the frame expands correctly
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    # creating drop downs to select some features of dataset
    font_all = ("Tahoma", 8, 'bold')
    #  label/result drop down
    Label(right_frame, bg=bg_right_frame, text="Enter the Label attribute :", font=font_all, width=29).grid(row=0, column=0, padx=(15,15), pady=(20,10), sticky='nw')
    options = list(df.columns)
    selected_option = StringVar() # Create a StringVar to hold the selected option
    selected_option.set(options[0])  # default value
    op=OptionMenu(right_frame, selected_option, *options)
    op.config(width=10)
    op.grid(row=0, column=1, padx=0, pady=(20,10), sticky='e')

    #  categorical or numerical
    Label(right_frame, bg=bg_right_frame, text="Categorical or Numerical :", font=font_all, width=29).grid(row=1, column=0, padx=(15,15), pady=(20,10), sticky='nw')
    cat_or_num = ['Categorical', 'Numerical']
    selected_option1 = StringVar()
    selected_option1.set(cat_or_num[0])
    op1=OptionMenu(right_frame, selected_option1, *cat_or_num)
    op1.config(width=10)
    op1.grid(row=1, column=1, padx=0, pady=(20,10), sticky='ne')

    #  algorithm to tackel imbalance dataset
    Label(right_frame, bg=bg_right_frame, text="Solve imbalance dataset :", font=font_all, width=29).grid(row=2, column=0, padx=(15,15), pady=(20,10), sticky='nw')
    im_problem = ['No Need','Undersampling', 'Oversampling', 'SMOTE']
    selected_option2 = StringVar()
    selected_option2.set(im_problem[0])
    op2=OptionMenu(right_frame, selected_option2, *im_problem)
    op2.config(width=10)
    op2.grid(row=2, column=1, padx=0, pady=(20,10), sticky='e')

    #  attributes for one hot encoding
    Label(right_frame, bg=bg_right_frame, text="Attributes for encoding :", font=font_all, width=29).grid(row=3, column=0, padx=(15,15), pady=(20,20), rowspan=2, sticky='nw')
    #   Here we need some kind of looping system to select multiple options from the drop down list
    #options = ['test_case1', 'test_case2', 'test_case3']
    selected_option3 = StringVar()
    selected_option3.set("No need")
    op3=OptionMenu(right_frame, selected_option3, *options)
    op3.config(width=10, height=1)
    op3.grid(row=3, column=1, padx=0, pady=(20,0), sticky='e')
    chooseMore_btn = Button(right_frame, text='Choose more', command=lambda: extract(4, selected_option3), font = ("Tahoma", 10), bg="#140358", fg='white')
    chooseMore_btn.config(width=13)
    chooseMore_btn.grid(row=4, column=1, padx=(0,0), pady=(0,10), sticky='e')

    # attributes for deletion
    Label(right_frame, bg=bg_right_frame, text="Drop a column :", font=font_all, width=29).grid(row=5, column=0, padx=(15,15), pady=(0,0), sticky='nw', rowspan=2)
    selected_option4 = StringVar()
    selected_option4.set('No need')
    op4=OptionMenu(right_frame, selected_option4, *options)
    op4.config(width=10)
    op4.grid(row=5, column=1, padx=0, pady=(0,0), sticky='e')
    chooseMore_btn1 = Button(right_frame, text='Choose more', command=lambda: extract(5, selected_option4), font = ("Tahoma", 10), bg="#140358", fg='white')
    chooseMore_btn1.config(width=13)
    chooseMore_btn1.grid(row=6, column=1, padx=(0,0), pady=(0,0), sticky='e')

    #  scaling techniques
    Label(right_frame, bg=bg_right_frame, text="Scaling methods :", font=font_all, width=29).grid(row=7, column=0, padx=(15,15), pady=(10,10), sticky='nw')
    scaling = ['No Need','MinMaxScaler', 'StandardScaler']
    selected_option5 = StringVar()
    selected_option5.set(scaling[0])
    op5=OptionMenu(right_frame, selected_option5, *scaling)
    op5.config(width=10)
    op5.grid(row=7, column=1, padx=0, pady=(10,10), sticky='e')

    

    # Last Lock button to lock all the selected options
    lock = Button(right_frame, text='LOCK ALL OPTIONS', bg="#8F76F1", fg='white', font=("Tahoma", 10, "bold"), command=lambda: (extract(1, selected_option), 
                                                                         extract(2, selected_option1),
                                                                         extract(3, selected_option2),
                                                                         extract(4, selected_option3),
                                                                         extract(5, selected_option4),
                                                                         extract(6, selected_option5),
                                                                         EDA(df, info_window)))
    lock.grid(row=8, column=0, padx=(80,55), pady=(60,0), columnspan=2, sticky='nsew')

    center_window(info_window)
    info_window.mainloop()

root = Tk() # object of class Tk()
root.overrideredirect(True)
root.geometry(f"{WIDTH}x{HEIGHT}")
root.resizable(False, False)  # Lock the size of the window

# Custom title bar
title_bar = Frame(root, bg="#251C47", bd=2)
title_bar.pack(fill=X)

title_label = Label(title_bar, text=" ", bg="#251C47", fg="white", font=("Helvetica", 12, "bold"))
title_label.pack(side=LEFT, padx=10)

# Close button
close_button = Button(title_bar, text="X", command=root.destroy, bg="#251C47", fg="white", borderwidth=0)
close_button.pack(side=RIGHT, padx=10)

background_img(root)

# write some text on the opening window
text = '''
                  Hey welcome
Insert any dataset of your choice
'''
open_win_label = Label(root, text = text, height=4, width=160, font=("Arial", 14,'bold'), bg="#8D83B7", fg="#0B0919", justify=LEFT)
#open_win_label.place(x=WIDTH//2 - 208, y=HEIGHT//2 - 200, pady = 10)
open_win_label.pack(padx = 5, pady = 15)

# widget accepting the file through its file path
entry_path = Entry(root, bd=6, width=40, font=("Tahoma", 11, 'bold'), bg = "#AE89F3", fg="#F8F8F8", justify = 'center')
entry_path.pack(side=TOP, pady=(90, 0), ipady=10)
entry_path.insert(0, "ENTER THE PATH") # Insert default text

# Button for accepting the entry in entry widget
click = Button(root, command = lambda: info_win(pd.read_csv(entry_path.get()), root),bd=3, height=1, width=15, text='Submit', font=(("Arial", 10,'bold')), bg = "#260A54", fg='#F8F8F8')
click.pack(side=TOP, pady=(10, 5), ipady=3)

center_window(root)
root.mainloop() # it constantly keeps the gui on screen