'''
Author: Zhou Hao
Date: 2021-09-05 18:41:38
LastEditors: Zhou Hao
LastEditTime: 2021-11-27 10:44:22
Description: the code of Icorss-validation framwork.
            5 Oversamplings algorithm are used.
E-mail: 2294776770@qq.com
'''
import warnings
warnings.filterwarnings("ignore")
from random import randrange, choice
import random
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn import datasets, metrics
from report import get_Gmean, Precision, get_Fmeature, TNR, TPR
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.utils import shuffle
import math
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,recall_score
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import StratifiedKFold
import os


def Smote(X, k:int):
    n_minority_samples, n_features = X.shape
    new_data = np.zeros(shape=(n_minority_samples, n_features))

    # Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)

    # Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors([X[i]], return_distance=False)[:, 1:]
        nn_index = choice(nn[0])
        dif = X[nn_index] - X[i]
        gap = np.random.uniform()
        new_data[i, :] = X[i, :] + gap * dif[:]
    return new_data


def BorderlineSMOTE(X, y, minority_target, k):
    n_samples, _ = X.shape
    # Learn nearest neighbours on complete training set
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)

    safe_minority_indices = list()
    danger_minority_indices = list()

    for i in range(n_samples):
        if y[i] != minority_target: continue
        nn = neigh.kneighbors([X[i]], return_distance=False)[:, 1:]
        majority_neighbours = 0

        for n in nn[0]:
            if y[n] != minority_target:
                majority_neighbours += 1

        if majority_neighbours == len(nn):
            continue
        elif majority_neighbours < (len(nn) / 2):
            safe_minority_indices.append(i)
        else:
            danger_minority_indices.append(i)

    synthetic_samples = Smote(X[danger_minority_indices], k)
    return synthetic_samples


class MDO(object):
    """
    Mahalanbois Distance Oversampling is an algorithm that oversamples all classes to a quantity of the major class.
    Samples for oversampling are chosen based on their k neighbours and new samples are created in random place but
    with the same Mahalanbois distance from the centre of class to chosen sample.
    """

    def __init__(self, k=5, k1_frac=.0, seed=0):
        self.knn = NearestNeighbors(n_neighbors=k)
        self.k2 = k
        self.k1 = int(k * k1_frac)
        self.random_state = check_random_state(seed)
        self.X, self.y = None, None

    def fit_transform(self, X, y, N):
        """
        Parameters
        ----------
        X two dimensional numpy array (number of samples x number of features) with float numbers
        y one dimensional numpy array with labels for rows in X
        Returns
        -------
        Resampled X and y
        """
        self.knn.fit(X)
        self.X, self.y = X, y

        quantities = Counter(y)
        goal_quantity = int(max(list(quantities.values())))

        class_label = 0
        chosen_minor_class_samples_to_oversample, weights = self._choose_samples(class_label)

        chosen_samples_mean = np.mean(chosen_minor_class_samples_to_oversample, axis=0)
        zero_mean_samples = chosen_minor_class_samples_to_oversample - chosen_samples_mean

        n_components = min(zero_mean_samples.shape)
        pca = PCA(n_components=n_components).fit(zero_mean_samples)

        uncorrelated_samples = pca.transform(zero_mean_samples)
        variables_variance = np.diag(np.cov(uncorrelated_samples, rowvar=False))
        oversampling_rate = int((goal_quantity - quantities[class_label]) / N)

        if oversampling_rate > 0:
            oversampled_set = self._MDO_oversampling(uncorrelated_samples, variables_variance, oversampling_rate,
                                                     weights)
            oversampled_set = pca.inverse_transform(oversampled_set) + chosen_samples_mean

        return oversampled_set

    def _choose_samples(self, class_label):
        minor_class_indices = [i for i, value in enumerate(self.y) if value == class_label]
        minor_set = self.X[minor_class_indices]

        quantity_same_class_neighbours = self.calculate_same_class_neighbour_quantities(minor_set, class_label)
        chosen_minor_class_samples_to_oversample = minor_set[quantity_same_class_neighbours >= self.k1]

        weights = quantity_same_class_neighbours[quantity_same_class_neighbours >= self.k1] / self.k2
        weights_sum = np.sum(weights)

        if weights_sum != 0:
            weights /= np.sum(weights)
        elif len(weights) > 0:
            value = 1 / len(weights)
            weights += value

        return chosen_minor_class_samples_to_oversample, weights

    def _MDO_oversampling(self, T, v, oversampling_rate, weights):
        oversampledSet = list()
        V = v + 1e-16

        for _ in range(oversampling_rate):
            idx = self.random_state.choice(np.arange(len(T)), p=weights)
            X = np.square(T[idx])
            a = np.sum(X / V)
            alpha_V = a * V

            s = 0
            features_vector = list()
            for alpha_V_j in alpha_V[:-1]:
                sqrt_avj = np.sqrt(alpha_V_j)
                r = self.random_state.uniform(low=-sqrt_avj, high=sqrt_avj)
                s += r ** 2 / sqrt_avj
                features_vector.append(r)

            last = (1 - s) * alpha_V[-1]
            last_feature = np.sqrt(last) if last > 0 else 0
            random_last_feature = self.random_state.choice([-last_feature, last_feature], 1)[0]

            features_vector.append(random_last_feature)
            oversampledSet.append(features_vector)

        return np.array(oversampledSet)

    def calculate_same_class_neighbour_quantities(self, S_minor, S_minor_label):
        minority_class_neighbours_indices = self.knn.kneighbors(S_minor, return_distance=False)
        quantity_with_same_label_in_neighbourhood = list()
        for i in range(len(S_minor)):
            sample_neighbours_indices = minority_class_neighbours_indices[i][1:]
            quantity_sample_neighbours_indices_with_same_label = sum(self.y[sample_neighbours_indices] == S_minor_label)
            quantity_with_same_label_in_neighbourhood.append(quantity_sample_neighbours_indices_with_same_label)
        return np.array(quantity_with_same_label_in_neighbourhood)


def ADASYN(X, Y, minority_target, G, K):
    "G:number to generate"

    synthetic = []
    r = []  # weights of the minority samplers
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(X)
    for i in range(0, len(Y)):
        if Y[i] == minority_target:
            delta = 0 
            nn = neigh.kneighbors([X[i]], K, return_distance=False)[:, 1:]
            neighbors = nn[0]
            for neighbors_index in neighbors:
                if Y[neighbors_index] != minority_target:
                    delta += 1
            r.append(1. * delta / K)   

    r = np.array(r)
    sum_r = np.sum(r)   
    r = r / sum_r       
    g = r * G       # The number of samples that need to be generated for each minority samples
    index = 0
    for i in range(0, len(Y)):  
        if Y[i] == minority_target:
            nn = neigh.kneighbors([X[i]], K, return_distance=False)[:, 1:]
            neighbors = nn[0]
            xzi_set = []
            for j in neighbors:
                if Y[j] == minority_target:
                    xzi_set.append(j)
            if len(xzi_set) == 0:
                continue

            for g_index in range(0, int(g[index])):     
                random_num = random.randint(0, len(xzi_set) - 1)
                xzi = np.array(X[xzi_set[random_num]])
                xi = np.array(X[i])
                random_lambda = random.random()
                synthetic.append((xi + (xzi - xi) * random_lambda).tolist())
            index += 1
    return np.array(synthetic)


def loadData(filename):
    # Read dataset from file
    df = pd.read_csv(r'dataset/%s' % filename, header=None,
                    encoding='gb2312')
    data = df.values
    x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.25, random_state=42)
    return x_train, y_train, x_test, y_test,data


def Stratified_5_fold(X,y,clf):
    g_mean = 0  # init g_mean

    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    for train_index, test_index in skf.split(X,y): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train,y_train.ravel())
        predictions = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        g_mean += math.sqrt(recall_score(y_test,predictions)*(tn/(tn+fp)))

    return round(g_mean/5,5)


def main():
    "save the results of each round"

    datasets = ['abalone', 'breastcancer', 'ecoli', 'glass', 'vowel', 'htru2', 'letter', 'yeast', 'newthyroid', 'poker', 'userknowledge']
    classifiers = ['GBDT', 'SVM', 'LR', 'DT', 'LGB']

    for dataset in datasets:
        print('\ndataset:\t',dataset)
        result = []
        result.append(
            ['Dataset', 'Clf', 'Methods', 'OA','Precision', 'Recall', 'F-measure', 'G-mean', 'Auc','train_Gmean', 'Rounds', 'Banlance Rate'])

        x_train, y_train, x_test, y_test = loadData(dataset)    

        df1 = pd.DataFrame(x_train, index=None)
        df2 = pd.DataFrame(y_train, index=None)
        df_train = pd.concat((df2, df1), axis=1, ignore_index=True)
        df_max = df_train[df_train[0] == 1]     #majority
        df_min = df_train[df_train[0] == 0]     #minority
        print('major:',len(df_max),'\tminor:',len(df_min))
        

        # choose Classifier
        for classifier in classifiers:
            if classifier == 'SVM':
                clf = SVC()
            elif classifier == 'DT':
                clf = DecisionTreeClassifier()
            elif classifier == 'LR':
                clf = LogisticRegression()
            elif classifier == 'GBDT':
                clf = GradientBoostingClassifier()
            elif classifier == 'LGB':
                clf = LGBMClassifier()
            print('\nclassifier:\t',classifier)


            "Smote: +5"
            N = int(df_max.shape[0] / df_min.shape[0]) * 1
            x_train1, y_train1 = x_train, y_train
            for j in range(1, N):
                x_new1 = Smote(df_min.iloc[:, 1:].values, 5)
                lenth = x_new1.shape[0]
                y_new1 = np.zeros(lenth)
                x_train1 = np.vstack((x_train1, x_new1))
                y_train1 = np.hstack((y_train1, y_new1))
                rate = len(df_max) / (len(x_train1) - len(df_max))  
                gmean = Stratified_5_fold(x_train1, y_train1,clf) 

                clf.fit(x_train1, y_train1)
                smotePred1 = clf.predict(x_test)
                Gmean = get_Gmean(y_test, smotePred1)
                precision = Precision(y_test, smotePred1)
                Fmeature = get_Fmeature(y_test, smotePred1, 1)
                recall = TPR(y_test, smotePred1)
                auc = roc_auc_score(y_test,smotePred1)
                print(dataset, classifier, j, 'Smote', gmean)
                result.append(
                    [dataset, classifier, 'Smote', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                    Fmeature, Gmean, auc, gmean, j, rate])


            "ADASYN"
            N = int(df_max.shape[0] / df_min.shape[0]) * 1
            x_train2, y_train2 = x_train, y_train
            for j in range(1,N+1):
                x_new1 = ADASYN(x_train, y_train,0,int(len(df_max)/N),5)
                lenth = x_new1.shape[0]
                y_new1 = np.zeros(lenth)
                x_train2 = np.vstack((x_train2, x_new1))
                y_train2 = np.hstack((y_train2, y_new1))
                rate = len(df_max) / (len(x_train2) - len(df_max))  
                gmean = Stratified_5_fold(x_train2, y_train2,clf) 

                
                clf.fit(x_train2, y_train2)
                smotePred1 = clf.predict(x_test)
                Gmean = get_Gmean(y_test, smotePred1)
                precision = Precision(y_test, smotePred1)
                Fmeature = get_Fmeature(y_test, smotePred1, 1)
                recall = TPR(y_test, smotePred1)
                auc = roc_auc_score(y_test,smotePred1)
                print(dataset, classifier, j, 'ADASYN', gmean)
                result.append(
                    [dataset, classifier, 'ADASYN', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                    Fmeature, Gmean, auc, gmean, j, rate])


            "BorderlineSmote:"
            x_train3, y_train3 = x_train, y_train
            j = 1
            while y_train3.shape[0] < 2 * df_max.shape[0]:
                x_new3 = BorderlineSMOTE(x_train, y_train, 0, 5)
                lenth = x_new3.shape[0]
                y_new3 = np.zeros(lenth)
                x_train3 = np.vstack((x_train3, x_new3))
                y_train3 = np.hstack((y_train3, y_new3))
                rate = len(df_max) / (len(x_train3) - len(df_max))
                gmean = Stratified_5_fold(x_train3, y_train3,clf)            

                clf.fit(x_train3, y_train3)
                smotePred1 = clf.predict(x_test)
                Gmean = get_Gmean(y_test, smotePred1)
                precision = Precision(y_test, smotePred1)
                Fmeature = get_Fmeature(y_test, smotePred1, 1)
                recall = TPR(y_test, smotePred1)
                auc = roc_auc_score(y_test,smotePred1)
                print(dataset, classifier, j, 'BorderlineSmote', gmean)
                result.append(
                    [dataset, classifier, 'BorderlineSmote', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                    Fmeature, Gmean, auc, gmean, j, rate])
                j += 1


            "MDO: +5"
            N = int(df_max.shape[0] / df_min.shape[0]) * 1
            x_train4, y_train4 = x_train, y_train
            for j in range(1, N):
                so = MDO()
                x_new4 = so.fit_transform(x_train, y_train, N)
                lenth = x_new4.shape[0]
                y_new4 = np.zeros(lenth)
                x_train4 = np.vstack((x_train4, x_new4))
                y_train4 = np.hstack((y_train4, y_new4))
                rate = len(df_max) / (len(x_train4) - len(df_max))
                gmean = Stratified_5_fold(x_train4, y_train4,clf) 

                clf.fit(x_train4, y_train4)
                smotePred1 = clf.predict(x_test)
                Gmean = get_Gmean(y_test, smotePred1)
                precision = Precision(y_test, smotePred1)
                Fmeature = get_Fmeature(y_test, smotePred1, 1)
                recall = TPR(y_test, smotePred1)
                auc = roc_auc_score(y_test,smotePred1)
                print(dataset, classifier, j, 'MDO', gmean)
                result.append(
                    [dataset, classifier, 'MDO', accuracy_score(y_test, smotePred1), precision, recall,
                    Fmeature, Gmean, auc, gmean, j, rate])


            "SVMSMOTE"
            x_train5, y_train5 = x_train, y_train
            N = int(df_max.shape[0] / df_min.shape[0]) * 1
            for j in range(1, N+1):
                so = SVMSMOTE(n_samples=int(len(df_max)/N))
                x_train5,y_train5 = so.fit_resample(x_train5, y_train5)
                rate = len(df_max) / (len(x_train5) - len(df_max))
                gmean = Stratified_5_fold(x_train5, y_train5,clf) 

                clf.fit(x_train5, y_train5)
                smotePred1 = clf.predict(x_test)
                Gmean = get_Gmean(y_test, smotePred1)
                precision = Precision(y_test, smotePred1)
                Fmeature = get_Fmeature(y_test, smotePred1, 1)
                recall = TPR(y_test, smotePred1)
                auc = roc_auc_score(y_test,smotePred1)
                print(dataset, classifier, j, 'SVMSMOTE', gmean)
                result.append(
                    [dataset, classifier, 'SVMSMOTE', accuracy_score(y_test, smotePred1), precision, recall,
                    Fmeature, Gmean, auc, gmean, j, rate])


        df = pd.DataFrame(result)
        df.to_csv(r'new_results/%s.csv' % dataset, header=None, index=None)
    print('\nOver')


def main_slim():
    "just save the highest gmean and the gmean in a balanced state"
    
    datasets = ['frogs','OBS','vehicle','vertebralColumn','pendigits','nuclear','contraceptive','seeds','shuttle','segmentation','sensorReadings']
    classifiers = ['GBDT', 'SVM', 'LR', 'DT', 'LGB']

    # for dataset in datasets:
    for dataset in os.listdir('dataset'):
        try:
            print('\ndataset:\t',dataset)
            result = []
            result.append(['Dataset', 'Clf', 'Methods', 'OA','Precision', 'Recall', 'F-measure', 'G-mean', 'Auc','train_Gmean', 'Rounds', 'Banlance Rate'])

            x_train, y_train, x_test, y_test,data = loadData(dataset)   
            X,y = data[:, 1:],data[:, 0]    
            print(set(y))
            if(len(set(y)) > 2):continue     #just binary
            
            df1 = pd.DataFrame(x_train, index=None)
            df2 = pd.DataFrame(y_train, index=None)
            df_train = pd.concat((df2, df1), axis=1, ignore_index=True)
            df_max = df_train[df_train[0] == 1]     # majority label: 1
            df_min = df_train[df_train[0] == 0]     # minority label: 0
            print('major:',len(df_max),'\tminor:',len(df_min),Counter(y))


            # choose Classifier
            for classifier in classifiers:
                if classifier == 'SVM':
                    clf = SVC()
                elif classifier == 'DT':
                    clf = DecisionTreeClassifier()
                elif classifier == 'LR':
                    clf = LogisticRegression()
                elif classifier == 'GBDT':
                    clf = GradientBoostingClassifier()
                elif classifier == 'LGB':
                    clf = LGBMClassifier()
                print('\nclassifier:\t',classifier)


                "Smote: +5"
                pre_train_gmean = 0
                N = int(df_max.shape[0] / df_min.shape[0]) * 1
                x_train1, y_train1 = x_train, y_train
                for j in range(1, N+1):
                    x_new1 = Smote(df_min.iloc[:, 1:].values, 5)
                    lenth = x_new1.shape[0]
                    y_new1 = np.zeros(lenth)
                    x_train1 = np.vstack((x_train1, x_new1))
                    y_train1 = np.hstack((y_train1, y_new1))

                    if len(df_max) < (len(x_train1) - len(df_max)):
                        x_train1 = x_train1[:2*len(df_max),:]
                        y_train1 = y_train1[:2*len(df_max)]
                    rate = len(df_max) / (len(x_train1) - len(df_max))          # (maj / total - maj) = maj / min
                    cur_train_gmean = Stratified_5_fold(x_train1, y_train1,clf) 

                    clf.fit(x_train1, y_train1)
                    smotePred1 = clf.predict(x_test)
                    Gmean = get_Gmean(y_test, smotePred1)
                    precision = Precision(y_test, smotePred1)
                    Fmeature = get_Fmeature(y_test, smotePred1, 1)
                    recall = TPR(y_test, smotePred1)
                    auc = roc_auc_score(y_test,smotePred1)
                    print(dataset, classifier, j, 'Smote', cur_train_gmean,round(rate,3))

                    if rate == 1 or j == 1: 
                        result.append(
                            [dataset, classifier, 'Smote', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate])
                    else:
                        if cur_train_gmean > pre_train_gmean :  
                            pre_train_gmean = cur_train_gmean
                            result[-1] = [dataset, classifier, 'Smote', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate]


                "ADASYN"
                pre_train_gmean = 0
                N = int(df_max.shape[0] / df_min.shape[0]) * 1
                j = 1
                x_train2, y_train2 = x_train, y_train
                while y_train2.shape[0] < 2 * df_max.shape[0]:
                    x_new1 = ADASYN(x_train, y_train,0,int(len(df_max)/N),5)
                    lenth = x_new1.shape[0]
                    y_new1 = np.zeros(lenth)

                    x_train2 = np.vstack((x_train2, x_new1))
                    y_train2 = np.hstack((y_train2, y_new1))
                    if len(df_max) < (len(x_train2) - len(df_max)):
                        x_train2 = x_train2[:2*len(df_max),:]
                        y_train2 = y_train2[:2*len(df_max)]
                    rate = len(df_max) / (len(x_train2) - len(df_max))  # (maj / total - maj) = maj / min
                    cur_train_gmean = Stratified_5_fold(x_train2, y_train2,clf) 
                    
                    clf.fit(x_train2, y_train2)
                    smotePred1 = clf.predict(x_test)
                    Gmean = get_Gmean(y_test, smotePred1)
                    precision = Precision(y_test, smotePred1)
                    Fmeature = get_Fmeature(y_test, smotePred1, 1)
                    recall = TPR(y_test, smotePred1)
                    auc = roc_auc_score(y_test,smotePred1)
                    print(dataset, classifier, j, 'ADASYN', cur_train_gmean,round(rate,3))

                    if rate == 1 or j == 1: 
                        result.append(
                            [dataset, classifier, 'ADASYN', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate])
                    else:
                        if cur_train_gmean > pre_train_gmean :  
                            pre_train_gmean = cur_train_gmean
                            result[-1] = [dataset, classifier, 'ADASYN', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate]
                    j+=1


                "BorderlineSmote:"
                pre_train_gmean = 0
                x_train3, y_train3 = x_train, y_train
                j = 1
                while y_train3.shape[0] < 2 * df_max.shape[0]:
                    x_new3 = BorderlineSMOTE(x_train, y_train, 0, 5)
                    lenth = x_new3.shape[0]
                    y_new3 = np.zeros(lenth)
                    x_train3 = np.vstack((x_train3, x_new3))
                    y_train3 = np.hstack((y_train3, y_new3))

                    if len(df_max) < (len(x_train3) - len(df_max)):
                        x_train3 = x_train3[:2*len(df_max),:]
                        y_train3 = y_train3[:2*len(df_max)]
                    rate = len(df_max) / (len(x_train3) - len(df_max))
                    cur_train_gmean = Stratified_5_fold(x_train3, y_train3,clf)               
                    
                    clf.fit(x_train3, y_train3)
                    smotePred1 = clf.predict(x_test)
                    Gmean = get_Gmean(y_test, smotePred1)
                    precision = Precision(y_test, smotePred1)
                    Fmeature = get_Fmeature(y_test, smotePred1, 1)
                    recall = TPR(y_test, smotePred1)
                    auc = roc_auc_score(y_test,smotePred1)
                    print(dataset, classifier, j, 'BorderlineSmote', cur_train_gmean,round(rate,3))
                    
                    if rate == 1 or j == 1: 
                        result.append(
                            [dataset, classifier, 'BorderlineSmote', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate])
                    else:
                        if cur_train_gmean > pre_train_gmean :  
                            pre_train_gmean = cur_train_gmean
                            result[-1] = [dataset, classifier, 'BorderlineSmote', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate]
                    j += 1


                # "MDO: +5"
                pre_train_gmean = 0
                N = int(df_max.shape[0] / df_min.shape[0]) * 1
                x_train4, y_train4 = x_train, y_train
                j = 1
                while y_train4.shape[0] < 2 * df_max.shape[0]:
                    so = MDO()
                    x_new4 = so.fit_transform(x_train, y_train, N)
                    lenth = x_new4.shape[0]
                    y_new4 = np.zeros(lenth)
                    x_train4 = np.vstack((x_train4, x_new4))
                    y_train4 = np.hstack((y_train4, y_new4))
                    if len(df_max) < (len(x_train4) - len(df_max)):
                        x_train4 = x_train4[:2*len(df_max),:]
                        y_train4 = y_train4[:2*len(df_max)]

                    
                    rate = len(df_max) / (len(x_train4) - len(df_max))
                    cur_train_gmean = Stratified_5_fold(x_train4, y_train4,clf) 
                    
                    clf.fit(x_train4, y_train4)
                    smotePred1 = clf.predict(x_test)
                    Gmean = get_Gmean(y_test, smotePred1)
                    precision = Precision(y_test, smotePred1)
                    Fmeature = get_Fmeature(y_test, smotePred1, 1)
                    recall = TPR(y_test, smotePred1)
                    auc = roc_auc_score(y_test,smotePred1)
                    print(dataset, classifier, j, 'MDO', cur_train_gmean,round(rate,3))

                    if rate == 1 or j == 1: 
                        result.append(
                            [dataset, classifier, 'MDO', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate])
                    else:
                        if cur_train_gmean > pre_train_gmean :  
                            pre_train_gmean = cur_train_gmean
                            result[-1] = [dataset, classifier, 'MDO', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate]
                    j += 1


                # "SVMSMOTE"
                pre_train_gmean = 0
                x_train5, y_train5 = x_train, y_train
                N = int(df_max.shape[0] / df_min.shape[0]) * 1
                j = 1
                while y_train5.shape[0] < 2 * df_max.shape[0]:
                    so = SVMSMOTE(n_samples=int(len(df_max)/N))
                    x_train5,y_train5 = so.fit_resample(x_train5, y_train5)

                    if len(df_max) < (len(x_train5) - len(df_max)):
                        x_train5 = x_train5[:2*len(df_max),:]
                        y_train5 = y_train5[:2*len(df_max)]
                    rate = len(df_max) / (len(x_train5) - len(df_max))
                    cur_train_gmean = Stratified_5_fold(x_train5, y_train5,clf) 

                    clf.fit(x_train5, y_train5)
                    smotePred1 = clf.predict(x_test)
                    Gmean = get_Gmean(y_test, smotePred1)
                    precision = Precision(y_test, smotePred1)
                    Fmeature = get_Fmeature(y_test, smotePred1, 1)
                    recall = TPR(y_test, smotePred1)
                    auc = roc_auc_score(y_test,smotePred1)
                    print(dataset, classifier, j, 'SVMSMOTE', cur_train_gmean,round(rate,3))

                    if rate == 1 or j == 1: 
                        result.append(
                            [dataset, classifier, 'SVMSMOTE', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate])
                    else:
                        if cur_train_gmean > pre_train_gmean :  
                            pre_train_gmean = cur_train_gmean
                            result[-1] = [dataset, classifier, 'SVMSMOTE', metrics.accuracy_score(y_test, smotePred1), precision, recall,
                            Fmeature, Gmean, auc, cur_train_gmean, j, rate]
                    j += 1


            df = pd.DataFrame(result)
            df.to_csv(r'new_results/%s.csv' % dataset, header=None, index=None)

        except Exception as e:
            continue

    infomation = pd.DataFrame(infos,columns=['#Dataset','#Samples','#Majority','#Minority','#Atribute','Minority description'],index = None)
    infomation.to_csv('data_info.csv',index= False)
    print('\nOver')


if __name__ == '__main__':
    main_slim()
    # main()


