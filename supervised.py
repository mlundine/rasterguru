import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def main(image, mask, classDict, saveFolder):
    """
    Performs DecisionTree, RandomForest, MLP,
    AdaBoost, GaussianNB, and QuadraticDiscriminant
    supervised classification.  Outputs classified image
    and a text file with assessment results.
    inputs:
    image: path to image for training and testing
    mask: annotated image
    classDict: {'class_string':value1, 'second_class_string':value2,...}
    saveFolder: directory to save to
    """
    image = cv2.imread(image_path)[:,:,0]
    mask = cv2.imread(mask_path)[:,:,0]

        

    ##assigning classes
    rows, cols= image.shape
    classes = classDict
    n_classes = len(classes)

    # create a color palette we will use to colorize the predictions later
    palette = np.uint8([[255, 255, 255],[0, 0, 0]])

    ##reshaping
    full=image.ravel()
    full=full.reshape((-1, 1))    #Set this guy aside for full prediction

    ### create a mask with the same dimensions
    Xtrain=np.dstack(image)
    Ylabel=np.dstack(mask)

    ### flatten
    data = Xtrain.ravel().reshape((-1,1))    
    label = Ylabel.ravel().reshape((-1,1)) 

    ### split the training data up so we can test later
    X_train, X_test, y_train, y_test = train_test_split(data, label.ravel(), test_size=0.20)


    ##classifiers
    classifiers = [
       # KNeighborsClassifier(3, algorithm='brute'),
        #SVC(kernel="linear",cache_size=7000, verbose=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, verbose=1,random_state=0),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
        ]
    classifierNames = [
                       #'KNeighbors',
                       #'Support Vector Machine',
                       'DecisionTree',
                       'RandomForest',
                       'MLPClassifier',
                       'AdaBoostClassifier',
                       'GaussianNB',
                       'QuadraticDiscriminantAnalysis',
                       ]

    def train_and_predict(classifier,
                          classifierName,
                          X_train, X_test,
                          y_train, y_test,
                          full,
                          rows, cols):
        start_time = time.time()
        #train and predict
        clf = classifier
        clf.fit(X_train, y_train)
        y_t = clf.predict(full)
        predicted=y_t.reshape(rows, cols,1)

        ##show result
        fig=plt.figure(figsize=(18, 16))
        plt.imshow(palette[predicted][:,:,0])
        plt.title(classifierName)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.savefig(os.path.join(saveFolder,classifierName+'.png'), dpi=300)

        ##assess results
        expected = y_test
        predicted = clf.predict(X_test)
        ##
        original_stdout = sys.stdout # Save a reference to the original standard output
        end_time = time.time()
        
        with open(os.path.join(saveFolder,classifierName+'.txt'), 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print("Classification report for classifier %s:\n%s\n"
                  % (clf, metrics.classification_report(expected, predicted)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
            print ('Accuracy Score :',accuracy_score(expected, predicted))
            print("Time (s): ", end_time - start_time)
            sys.stdout = original_stdout # Reset the standard output to its original value

    ##Loop over various classifiers
    i=0
    for classifier in classifiers:
        
        train_and_predict(classifier,
                          classifierNames[i],
                          X_train, X_test,
                          y_train, y_test,
                          full,
                          rows,cols)
        i=i+1

