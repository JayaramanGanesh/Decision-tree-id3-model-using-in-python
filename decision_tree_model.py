import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus


#dataset loading
col_names = ["feature name mention array type"]
pima = pd.read_csv("dataset file with csv format", header=None, names=col_names)
pima.head()

#split the data set
feature_cols = ["dataset in features and target variable mention array type"]
X = pima[feature_cols] 
y = pima.label 

#training the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#loading the model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#accuracy checking
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#inside model chart printing
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


#Optimizing Decision Tree Performance
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Visualizing Decision Trees after optimizing Decision Tree Performance
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
