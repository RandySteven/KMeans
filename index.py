import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef
from scipy.spatial.distance import euclidean

sensus = {
    'tinggi':[158, 170, 183, 191, 155, 163, 180, 158, 178],
    'berat':[64,86,84,80,49,59,67,54,67],
    'jk':[
        'pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita', 'wanita'
    ]
}

#tinggi & berat pictures
#jk target

sensus_df = pd.DataFrame(sensus)
print(sensus_df)


fig, ax = plt.subplots()
for jk, d in sensus_df.groupby('jk'):
    ax.scatter(d['tinggi'], d['berat'], label=jk)

plt.legend(loc='upper left')
plt.title('Sebaran data tinggi dan berat dan jenis kelamin')
plt.xlabel('Tinggi badan (cm)')
plt.ylabel('Berat badan (kg)')
plt.grid(True)
plt.show()

#Classification KNN
X_train = np.array(sensus_df[['tinggi', 'berat']])
Y_train = np.array(sensus_df['jk'])

print(f'X_train: {X_train}')
print(f'Y_train: {Y_train}')

lb = LabelBinarizer()
Y_train = lb.fit_transform(Y_train)
print(f'Y_train: {Y_train}')

Y_train = Y_train.flatten()
print(Y_train)


K = 3
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, Y_train)

tinggi_badan = 155
berat_badan = 70
X_new = np.array([tinggi_badan, berat_badan]).reshape(1, -1)
# X_new

Y_new = model.predict(X_new)
# print(Y_new)

print(lb.inverse_transform(Y_new))

fig, ax = plt.subplots()
for jk, d in sensus_df.groupby('jk'):
    ax.scatter(d['tinggi'], d['berat'], label=jk)

plt.scatter(tinggi_badan, berat_badan, marker='s', color='red', label='misterius')
plt.legend(loc='upper left')
plt.title('Sebaran data tinggi dan berat dan jenis kelamin')
plt.xlabel('Tinggi badan (cm)')
plt.ylabel('Berat badan (kg)')
plt.grid(True)
plt.show()

misterius = np.array([tinggi_badan, berat_badan])
print(misterius)
print(X_train)

data_jarak = [euclidean(misterius, d) for d in X_train]
print(data_jarak)
sensus_df['jarak'] = data_jarak
print(sensus_df.sort_values(['jarak']))


#testing set
X_test = np.array([[168, 65], [180, 96], [160, 52], [169, 67]])
Y_test = lb.transform(np.array(['pria', 'pria', 'wanita', 'wanita'])).flatten()

print(f'X_test: {X_test}')
print(f'Y_test: {Y_test}')

#Prediksi terhadap testing set
Y_pred = model.predict(X_test)
print(Y_pred)


#accuracy = (true_positif + true_negatif) / (true_positif + true_negatif + false_positif + false_negatif)
acc = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {acc}')

#Precision = tp / (tp + fp)
prec = precision_score(Y_test, Y_pred)
print(f'Precision: {prec}')

#Recall = tp / (tp + fn)
rec = recall_score(Y_test, Y_pred)
print(f'Recall: {rec}')

#F1Score = 2 x (precission x recall)/(precission + recall)
f1 = f1_score(Y_test, Y_pred)
print(f'F1: {f1}')


#Classification REport
cls_report = classification_report(Y_test, Y_pred)
print(f'Classification Report: \n{cls_report}')


#MCC = (tp x tn + fp x fn) / ^(tp+fp) x (tp+fn) x (tn+fp) x (tn+fn)
mcc = matthews_corrcoef(Y_test, Y_pred)
print(f'MCC: {mcc}')