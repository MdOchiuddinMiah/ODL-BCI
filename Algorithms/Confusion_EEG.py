import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Utilities
def showPerformance(x_train, y_train, model):
    print('Accuracy of the training set: {:.2f}'.format(model.score(x_train, y_train) * 100) + ' %')
    print('Accuracy of the test set: {:.2f}'.format(model.score(x_test, y_test) * 100) + ' %')


class DataRetrive:
    firstFilePath = "/Volumes/NDSU/Research Study/BCI Confused Student Deep Learning/DataSet/archive/EEG_data.csv"  # eeg data
    secondFilePath = "/Volumes/NDSU/Research Study/BCI Confused Student Deep Learning/DataSet/archive/demographic_info.csv"  # data info
    class_col = "ClassLabel"
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def __init__(self, localPath, class_col):
        self.localPath = localPath
        self.class_col = class_col

    @classmethod
    def getFirstPath(cls):
        return cls.firstFilePath

    @classmethod
    def getSecondPath(cls):
        return cls.secondFilePath

    @classmethod
    def getClassCol(cls):
        return cls.class_col

    def getData(self, path):
        return pd.read_csv(path)

    def mergeData(self, mergeData, mainData):
        mergeData.rename(columns={'subject ID': 'SubjectID'}, inplace=True)
        return mergeData.merge(mainData, on='SubjectID')

    def dropCols(self, data, cols):
        return data.drop(cols, axis=1)

    def print_unique_col_values(df):
        for column in df:
            if df[column].dtypes == 'object':
                print(f'{column}: {df[column].unique()}')


class DataPreprocessing:

    def __init__(self, data):
        self.data = data

    def dataProcess(self):
        data = self.data
        data.rename(
            columns={' age': 'Age', ' ethnicity': 'Ethnicity', ' gender': 'Gender', 'user-definedlabeln': 'ClassLabel'},
            inplace=True)
        data['ClassLabel'] = data['ClassLabel'].astype(np.int)

        print(data)
        print("Missing values:", data.isna().sum().sum())

        data['Gender'].unique()
        data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

        data['Ethnicity'].unique()
        ethnicity_dummies = pd.get_dummies(data['Ethnicity'])
        data = pd.concat([data, ethnicity_dummies], axis=1)
        data = data.drop('Ethnicity', axis=1)
        self.data = data

        return data

    def dataSplit(self, continuous_features):
        y = self.data['ClassLabel'].copy()
        x = self.data.drop('ClassLabel', axis=1).copy()

        scaler = MinMaxScaler()
        x[continuous_features] = scaler.fit_transform(x[continuous_features])

        print('X columns types:')
        for col in x:
            print(f'{col}: {x[col].unique()}')

        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        return x_train, x_test, y_train, y_test

    def representData(self):

        features[continuous_features].plot(kind='box', figsize=(15, 15), subplots=True, layout=(3, 4))
        plt.show()

        features[continuous_features].plot(kind='hist', bins=25, figsize=(15, 12), subplots=True, layout=(3, 4))
        plt.show()

        plt.figure(figsize=(20, 5))
        for feature in categorical_features:
            plt.subplot(1, 5, categorical_features.index(feature) + 1)
            features[feature].value_counts().plot(kind='pie')
        plt.show()

        plt.figure(figsize=(8, 8))
        data['ClassLabel'].value_counts().plot(kind='pie', autopct='%.1f%%')
        plt.show()

        # Multivariate Analysis
        plt.figure(figsize=(20, 20))
        sns.pairplot(features[continuous_features])
        plt.show()

        corr = data.corr()

        plt.figure(figsize=(18, 15))
        sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
        plt.show()


class DeepModelPrepare:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def prepareDeepModel(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(self.x_train.shape[1],), activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(2, activation='relu'),
            # tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # return tf.keras.Sequential([
        #     tf.keras.layers.Dense(200, input_shape=(self.x_train.shape[1],), activation='relu'),
        #     tf.keras.layers.Dense(100, activation='relu'),
        #     tf.keras.layers.Dense(50, activation='relu'),
        #     tf.keras.layers.Dense(16, activation='relu'),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])
        # return tf.keras.Sequential([
        #     tf.keras.layers.Dense(64, input_shape=(x_train.shape[1],), activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.27),
        #     tf.keras.layers.Dense(124, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Dense(248, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.32),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.27),
        #     tf.keras.layers.Dense(664, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.32),
        #     tf.keras.layers.Dense(264, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.27),
        #     tf.keras.layers.Dense(124, activation='relu'),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dropout(0.3),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ])

    def modelEvaluate(self):
        plt.figure(figsize=(16, 10))

        plt.plot(range(epochs), history.history['loss'], label="Training Loss")
        plt.plot(range(epochs), history.history['val_loss'], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")
        plt.legend(loc="lower right")
        plt.show()

        plt.plot(range(epochs), history.history['accuracy'], label="Training Accuracy")
        plt.plot(range(epochs), history.history['val_accuracy'], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Time")
        plt.legend(loc="lower right")
        plt.show()

        print('Deep Model:')
        deepModel.evaluate(x_test, y_test)
        y_true = np.array(y_test)
        yp = deepModel.predict(x_test)

        y_pred = []
        for element in yp:
            if element >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_pred = np.squeeze(deepModel.predict(x_test))
        y_pred = np.array(y_pred >= 0.5, dtype=np.int)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        print(classification_report(y_true, y_pred))


class MLModelPrepare:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def showPerformance(self, clf, modelName):
        print(modelName)
        print('Accuracy of the training set: {:.2f}'.format(clf.score(self.x_train, self.y_train) * 100) + ' %')
        print('Accuracy of the test set: {:.2f}'.format(clf.score(self.x_test, self.y_test) * 100) + ' %')
        predicted = clf.predict(self.x_test)
        print(classification_report(y_test, predicted))

    def fitDecisionTree(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.x_train, self.y_train)
        return clf

    def fitAdaBoost(self):
        ada = AdaBoostClassifier(n_estimators=100, random_state=0)
        ada = ada.fit(self.x_train, self.y_train)
        return ada

    def fitBagging(self):
        bag = BaggingClassifier()
        bag = bag.fit(self.x_train, self.y_train)
        return bag

    def fitMLP(self):
        mlp = MLPClassifier(random_state=42)
        mlp = mlp.fit(self.x_train, self.y_train)
        return mlp

    def fitNaiveBayes(self):
        nb = MultinomialNB()
        nb = nb.fit(self.x_train, self.y_train)
        return nb

    def fitRandomForest(self):
        rf = RandomForestClassifier()
        rf = rf.fit(self.x_train, self.y_train)
        return rf

    def fitSVM(self):
        sv = svm.SVC(kernel='linear', probability=True)
        sv = sv.fit(self.x_train, self.y_train)
        return sv

    def fitXGBoost(self):
        xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, seed=1)
        xg = xg.fit(self.x_train, self.y_train)
        return xg


# call class
dataRetrive = DataRetrive(DataRetrive.getFirstPath(), DataRetrive.getClassCol())
eeg_df = dataRetrive.getData(DataRetrive.getFirstPath())
info_df = dataRetrive.getData(DataRetrive.getSecondPath())
data = dataRetrive.mergeData(info_df, eeg_df)
print(data.info())

data = dataRetrive.dropCols(data, ['SubjectID', 'VideoID', 'predefinedlabel'])
print(data.columns)

dataProcessing = DataPreprocessing(data)
data = dataProcessing.dataProcess()

print(data)
print("Non-numeric columns:", len(data.select_dtypes('object').columns))
print(data.dtypes)

features = data.drop('ClassLabel', axis=1).copy()
num_features = len(features.columns)
print("Features:", num_features)

categorical_features = ['Gender', 'Bengali', 'English', 'Han Chinese']
continuous_features = ['Age', 'Attention', 'Mediation', 'Raw', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2',
                       'Gamma1', 'Gamma2']

print("Categorical Features:", len(categorical_features))
print("Continuous Features:", len(continuous_features))

# Charts
# dataProcessing.representData()

x_train, x_test, y_train, y_test = dataProcessing.dataSplit(continuous_features)

print(x_train.shape[1])

batch_size = 28
epochs = 2000
modelPrepare = DeepModelPrepare(x_train, x_test, y_train, y_test)
deepModel = modelPrepare.prepareDeepModel()

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
deepModel.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

history = deepModel.fit(
    x_train,
    y_train,
    validation_data=(x_train, y_train),
    # batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau()
    ]
)

modelPrepare.modelEvaluate()
# ROC for DL
probs = deepModel.predict(x_test)
# probs = probs[:, 1]
y = y_test
auc_dl = roc_auc_score(y, probs)
print('AUC Deep Learning: %.3f' % auc_dl)
fpr_dl, tpr_dl, thresholds_dl = roc_curve(y, probs)

# ML Classifier
#
mlModelPrepare = MLModelPrepare(x_train, x_test, y_train, y_test)

dt = mlModelPrepare.fitDecisionTree()
mlModelPrepare.showPerformance(dt, 'Decision Tree')
# ROC for DT
probs = dt.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_dt = roc_auc_score(y, probs)
print('AUC Decision Tree: %.3f' % auc_dt)
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y, probs)

ada = mlModelPrepare.fitAdaBoost()
mlModelPrepare.showPerformance(ada, 'AdaBoost')
# ROC for AdaBoost
probs = ada.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_ada = roc_auc_score(y, probs)
print('AUC AdaBoost: %.3f' % auc_ada)
fpr_ada, tpr_ada, thresholds_ada = roc_curve(y, probs)

bag = mlModelPrepare.fitBagging()
mlModelPrepare.showPerformance(bag, 'Bagging')
# ROC for Bagging
probs = bag.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_bag = roc_auc_score(y, probs)
print('AUC Bagging: %.3f' % auc_bag)
fpr_bag, tpr_bag, thresholds_bag = roc_curve(y, probs)

mlp = mlModelPrepare.fitMLP()
mlModelPrepare.showPerformance(mlp, 'MLP')
# ROC for MLP
probs = mlp.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_mlp = roc_auc_score(y, probs)
print('AUC MLP: %.3f' % auc_mlp)
fpr_mlp, tpr_mlp, thresholds_mllp = roc_curve(y, probs)

nb = mlModelPrepare.fitNaiveBayes()
mlModelPrepare.showPerformance(nb, 'Naive Bayes')
# ROC for Naive Bayes
probs = nb.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_nb = roc_auc_score(y, probs)
print('AUC Naive Bayes: %.3f' % auc_nb)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y, probs)

rf = mlModelPrepare.fitRandomForest()
mlModelPrepare.showPerformance(rf, 'Random Forest')
# ROC for Random Forest
probs = rf.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_rf = roc_auc_score(y, probs)
print('AUC Random Forest: %.3f' % auc_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y, probs)

svm = mlModelPrepare.fitSVM()
mlModelPrepare.showPerformance(svm, 'Support Vector Machine')
# ROC for Support Vector Machine
probs = svm.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_svm = roc_auc_score(y, probs)
print('AUC Support Vector Machine: %.3f' % auc_svm)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y, probs)

xgBoost = mlModelPrepare.fitXGBoost()
mlModelPrepare.showPerformance(xgBoost, 'XG Boost')
# ROC for XG Boost
probs = xgBoost.predict_proba(x_test)
probs = probs[:, 1]
y = y_test
auc_xgBoost = roc_auc_score(y, probs)
print('AUC XG Boost: %.3f' % auc_xgBoost)
fpr_xgBoost, tpr_xgBoost, thresholds_xgBoost = roc_curve(y, probs)

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Area under ROC curve')

plt.plot(fpr_dt, tpr_dt, color='cyan', lw=2, label='DT (AUC: %0.2f)' % auc_dt)
plt.plot(fpr_ada, tpr_ada, color='green', lw=2, label='AdaBoost (AUC: %0.2f)' % auc_ada)
plt.plot(fpr_bag, tpr_bag, color='black', lw=2, label='Bagging (AUC: %0.2f)' % auc_bag)
plt.plot(fpr_mlp, tpr_mlp, color='red', lw=2, label='MLP (AUC: %0.2f)' % auc_mlp)
plt.plot(fpr_nb, tpr_nb, color='yellow', lw=2, label='NB Classifier (AUC: %0.2f)' % auc_nb)
plt.plot(fpr_dl, tpr_dl, color='magenta', lw=2, label='Random Forest (AUC: %0.2f)' % auc_dl)
plt.plot(fpr_svm, tpr_svm, color='orange', lw=2, label='SVM (AUC: %0.2f)' % auc_svm)
plt.plot(fpr_xgBoost, tpr_xgBoost, color='brown', lw=2, label='XG Boost (AUC: %0.2f)' % auc_xgBoost)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='ODL-BCI (AUC: %0.2f)' % auc_rf)

plt.legend(loc="lower right")
plt.show()
print('end roc')
