import pandas
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.ensemble


def load(file_csv):
    return pandas.read_csv(file_csv)


def describe_columns(dataset):
    for (columnName, columnData) in dataset.iteritems():
        print(columnName)
        df = pandas.Series(columnData)
        print(df.describe())
        print()


def plot(data, type='scatter'):
    for (columnName, columnData) in data.iteritems():
        if type == 'scatter':
            data.plot.scatter(y='classification', x=columnName)
        elif type == 'box':
            data.boxplot(column=columnName, by='classification')
        #plt.show()

        if columnName == "total_fpktl":
            plt.ylim(0, 600)
        elif columnName == "max_flowpktl":
            plt.ylim(0, 500)
        elif columnName == "mean_flowpktl":
            plt.ylim(0, 450)

        plt.savefig(type+'_'+columnName+'.png')
        plt.close()


def feature_evaluation(dataset, print_enable=False):
    features = []
    for (columnName, columnData) in dataset.iteritems():
        if columnName != 'classification':
            features.append(columnData)
        else:
            target = columnData

    X = np.array(features)
    X = X.transpose()
    Y = np.array(target)
    feature_selcted = sklearn.feature_selection.mutual_info_classif(X, Y, discrete_features=False)
    feature_selcted_zip = sorted(zip(feature_selcted, dataset.columns), reverse=True)

    if print_enable:
        for score, f_name in feature_selcted_zip:
            print(f_name, score)
        print(feature_selcted_zip)

    return feature_selcted_zip


def topIndipendentAttributeSelect(features_selected, n):
    topIndipendentAttribute = dict()
    for i in range(0, n):
        score, f_name = zip(*features_selected)
        topIndipendentAttribute[f_name[i]] = score[i]

    return topIndipendentAttribute

def PCA(data, n_componets):
    ## Select with PCA
    pca = sklearn.decomposition.PCA(n_components=n_componets)
    dataset_without_class = data.iloc[:, :-1]
    pca.fit(dataset_without_class)
    transform = pca.transform(dataset_without_class)
    dataset_only_class = np.asmatrix(data.iloc[:, -1]).T
    new_dataset = np.hstack((transform, dataset_only_class))

    return new_dataset, pca


def apply_PCA(data, pca):
    dataset_without_class = data.iloc[:, :-1]
    transform = pca.transform(dataset_without_class)
    dataset_only_class = np.asmatrix(data.iloc[:, -1]).T
    new_dataset = np.hstack((transform, dataset_only_class))

    return new_dataset


def stratified_cross_validation(data, n_splits, random=None):
    X=data.iloc[:, :-1]
    Y=data['classification']
    scv = sklearn.model_selection.StratifiedKFold(n_splits, shuffle=True, random_state=random)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for train_index, test_index in scv.split(X, Y):
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        Y_train.append(Y.iloc[train_index])
        Y_test.append(Y.iloc[test_index])

    return X_train, X_test, Y_train, Y_test


def random_forest_learner(X, Y, number_features="sqrt", number_samples=0.5, number_trees=25):
    rfc = sklearn.ensemble.RandomForestClassifier(max_features=number_features, max_samples=number_samples, n_estimators=number_trees, min_samples_split=0.05)
    rfc.fit(X, Y)

    return rfc

def evaluate_random_forest(X, Y_true, rfc):
    y_pred = rfc.predict(X)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import balanced_accuracy_score

    metrics = []
    metrics.append(accuracy_score(Y_true, y_pred))  # oa
    metrics.append(balanced_accuracy_score(Y_true, y_pred))  # balanced accuracy
    metrics.append(precision_score(Y_true, y_pred))  # precision
    metrics.append(recall_score(Y_true, y_pred))  # recall
    metrics.append(f1_score(Y_true, y_pred))  # fscore

    return metrics


def evaluateCV(X_Train, X_Test, Y_Train, Y_Test, number_features="sqrt", number_samples=0.5, number_trees=25):
    number_folds = len(X_Train)
    avgTrain = [0] * 5
    avgTest = [0] * 5

    for i in range(0, number_folds):
        rf = random_forest_learner(X_Train[i], Y_Train[i], number_features=number_features, number_samples=number_samples, number_trees=number_trees)
        scoreTrain = evaluate_random_forest(X_Train[i], Y_Train[i], rf)
        for j in range(0, len(avgTrain)):
            avgTrain[j] += scoreTrain[j]
        scoreTest = evaluate_random_forest(X_Test[i], Y_Test[i], rf)
        for j in range(0, len(avgTest)):
            avgTest[j] += scoreTest[j]

    for j in range(0, len(avgTrain)):
        avgTrain[j] = avgTrain[j]/number_folds

    for j in range(0, len(avgTest)):
        avgTest[j] = avgTest[j]/number_folds

    return avgTrain, avgTest



file = 'Train_OneClsNumeric.csv'
dataset = load(file)
# Print info
# print(dataset.shape)
# print(dataset.head())
# print(dataset.columns)

# Print info of each columns
#describe_columns(dataset)

#View plot of each column
#plot(dataset, type='box')


## Select Att with mult info
#feature_selcted = feature_evaluation(dataset, False)

#top_attribute = topIndipendentAttributeSelect(feature_selcted, 10)

#datasetWithTopAtributes = dataset.loc[:, top_attribute.keys()]

#print(datasetWithTopAtributes)

# dataset_PCA, pca = PCA(dataset, 5)

X_train, X_test, Y_train, Y_test = stratified_cross_validation(dataset, 5, 50)

#avg_train, avg_test = evaluateCV(X_train, X_test, Y_train, Y_test, number_features="sqrt", number_samples=0.9, number_trees=30)

#print("AVG Train")
#print(avg_train)
#print("AVG Test")
#print(avg_test)

randomizations = ["sqrt", "log2"]
bootstraps = [0.5, 0.6, 0.7, 0.8, 0.9]
numbers_of_trees = [10, 20, 30]

best_configuration = ["", 0.0, 0]
f_score_test_best = 0

for randomization in randomizations:
    for number_samples in bootstraps:
        for number_of_tree in numbers_of_trees:
            avg_train, avg_test = evaluateCV(X_train, X_test, Y_train, Y_test, number_features=randomization,
                                             number_samples=number_samples, number_trees=number_of_tree)
            f_score_test = avg_test[4]
            if f_score_test > f_score_test_best:
                f_score_test_best = f_score_test
                best_configuration = [randomization, number_samples, number_of_tree]

print("Best configuration: Randomization: "+best_configuration[0]+" Number of samples: "+str(best_configuration[1]) +
      " Number of trees: "+str(best_configuration[2]))

#for i in range(0, len(X_train)):
#    rfl = random_forest_learner(X_train[i], Y_train[i], number_features="sqrt", number_samples=0.9, number_trees=30)

