import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub
#import joblib

dagshub.init(repo_owner='SirIsaac96', repo_name='mlflow-exp-dagshub', mlflow=True)

mlflow.set_experiment("water-exp2")
mlflow.set_tracking_uri("https://dagshub.com/SirIsaac96/mlflow-exp-dagshub.mlflow")
data = pd.read_csv("C:/exp_mlflow/data/water_potability.csv")

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def handle_missing_values(data):
    for column in data.columns:
        if data[column].isnull().any():
            median_value = data[column].median()
            data[column].fillna(median_value, inplace=True)
    return data

train_processed_data = handle_missing_values(train_data)
test_processed_data = handle_missing_values(test_data)

from sklearn.ensemble import GradientBoostingClassifier
import pickle
x_train = train_processed_data.iloc[:, 0:-1].values
y_train = train_processed_data.iloc[:, -1].values

n_estimators = 1000

with mlflow.start_run():

    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(x_train, y_train)

    mlflow.sklearn.save_model(sk_model=clf, path="GradientBoostingModel")
    mlflow.log_artifact("GradientBoostingModel")

    # Save the trained model
    pickle.dump(clf, open('model_gb.pkl', 'wb'))

    x_test = test_processed_data.iloc[:, 0:-1].values
    y_test = test_processed_data.iloc[:, -1].values

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    model = pickle.load(open('model_gb.pkl', 'rb'))

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_param("n_estimators", n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('gb_confusion_matrix.png')

    mlflow.log_artifact('gb_confusion_matrix.png')

    # mlflow.sklearn.log_model(clf, "GraduentBoostingModel")

    # Log this script
    try:
        mlflow.log_artifact(__file__)
    except Exception:
        # __file__ may not be available in some execution contexts (e.g. interactive shells)
        pass

    mlflow.set_tag("author", "Isaac-Otom")
    mlflow.set_tag("model", "GB")

    print("accuracy", accuracy)
    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1)