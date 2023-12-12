import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("D:\\Computer-Vision\\Classification & Regression\\heart.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    'Stochastic Gradient Descent': SGDClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Machines': SVC()
}

# Scaling features if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


results_table = pd.DataFrame(columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1'])

# Perform five-fold cross-validation
for method, classifier in classifiers.items():
    scores_accuracy = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
    scores_precision = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='precision')
    scores_recall = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='recall')
    scores_f1 = cross_val_score(classifier, X_train_scaled, y_train, cv=5, scoring='f1')

    results_table = results_table.append({
        'Method': method,
        'Accuracy': scores_accuracy.mean(),
        'Precision': scores_precision.mean(),
        'Recall': scores_recall.mean(),
        'F1': scores_f1.mean()
    }, ignore_index=True)

# Compare the models on the test set
for method, classifier in classifiers.items():
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_table = results_table.append({
        'Method': method + ' (Test)',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }, ignore_index=True)

print(results_table)
