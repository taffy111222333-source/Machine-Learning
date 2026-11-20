import seaborn as sns
import numpy as np
import sklearn.preprocessing
from sklearn import datasets, model_selection, linear_model
'''提取数据'''
data = datasets.load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
'''标准化'''
scaler = sklearn.preprocessing.StandardScaler()
X_train_standard = scaler.fit_transform(X_train)
X_test_standard = scaler.transform(X_test)

'''拟合模型'''

model = linear_model.LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

'''评价模型'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import f1_score, classification_report

print(classification_report(y_test, y_pred))
print('Predicted labels: ', y_pred)
print('F1-score: ', f1_score(y_test, y_pred))
print(cm)