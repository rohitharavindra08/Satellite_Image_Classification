import numpy as np
from skimage import io
from PIL import Image
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

img = io.imread('TFOv7.png')
d = Image.fromarray(img)

rows, cols, bands = img.shape
classes = {'building': 0, 'vegetation': 1, 'water': 2}
n_classes = len(classes)
palette = np.uint8([[255, 0, 0], [0, 255, 0], [0, 0, 255]])


supervised = n_classes*np.ones(shape=(rows, cols), dtype=int)
supervised[200:220, 150:170] = classes['building']
supervised[40:60, 40:60] = classes['vegetation']
supervised[100:120, 200:220] = classes['water']
supervised[100:120, 220:240] = classes['water']
supervised[70:100, 240:260] = classes['water']
supervised[150:170, 220:240] = classes['water']
supervised[130:150, 60:80] = classes['water']
supervised[80:105,230:250] = classes['water']


X = img.reshape(rows*cols, bands)
y = supervised.ravel()

train = np.flatnonzero(supervised < n_classes)
test = np.flatnonzero(supervised == n_classes)

def Kmeans(n_classes,X):
    kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
    unsupervised = kmeans.labels_.reshape(rows, cols)
    k_m = Image.fromarray(palette[unsupervised])
    k_m.show()

def _SVM_(X,y,train,test):
    sv = SVC(gamma='auto')
    sv.fit(X[train], y[train])
    y[test] = sv.predict(X[test])
    supervised_sv = y.reshape(rows, cols)
    svm = Image.fromarray(palette[supervised_sv])
    svm.show()
    print("/***Accuracy score of SVM***\\")
    print("{:.2f}".format(sv.score(X[train], y[train])*100))

def Random_Forest(X,y,train,test):
    random_forest = RandomForestClassifier()
    random_forest.fit(X[train], y[train])
    y[test] = random_forest.predict(X[test])
    supervised_rf = y.reshape(rows, cols)
    rf = Image.fromarray(palette[supervised_rf])
    rf.show()
    print("/***Accuracy score of Random forest***\\")
    print("{:.2f}".format(random_forest.score(X[train], y[train])*100))

def Decision_Tree(X,y,train,test):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X[train], y[train])
    y[test] = decision_tree.predict(X[test])
    supervised_dt = y.reshape(rows, cols)
    dt = Image.fromarray(palette[supervised_dt])
    dt.show()
    print("/***Accuracy score of Decision Tree***\\")
    print("{:.2f}".format(decision_tree.score(X[train], y[train])*100))

'''
Kmeans(n_classes,X)
_SVM_(X,y,train,test)
Random_Forest(X,y,train,test)
Decision_Tree(X,y,train,test
'''

print(rows,cols)