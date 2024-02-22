# This is a Python script that verifies the anonimity of a health public dataset.
# Adrian Serrano Navarro

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from performance_calculation import performance_rates
import numpy as np

#parametro de medida de k-anonimidad
k=5

dataset = pd.read_excel("dataset\\Dns2009_StrongPasswordData.xls")
dataframe = pd.DataFrame(dataset)

casos =["Normal", "ruido 0 y 0,1", "ruido 0,1 y 0,1", "generalización con redondeo"]
for i in range(0, 4):
    data = dataframe.drop(["subject", "sessionIndex", "rep"], axis="columns")
    labels = dataframe.subject

    if i == 1:
        data += np.random.normal(loc=0, scale=0.1, size=data.shape)

    if i == 2:
        data += np.random.normal(loc=0.1, scale=0.1, size=data.shape)

    if i == 3:
        data = data.round(2)

    #k-anonimity
    ncaract_noanonimas = 0
    for j in data.columns:
        if data[j].nunique() < k:
            ncaract_noanonimas += 1

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    random_forest = RandomForestClassifier(criterion="entropy",
                                           max_features="sqrt",
                                           max_depth=17,
                                           n_estimators=225)

    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)

    # quality of results
    rates = performance_rates(y_test, y_pred)
    f1 = rates[0]
    accuracy = rates[1]
    false_positive = rates[2]
    false_negative = rates[3]

    print("Caso ", casos[i])
    print("Numero de características es", data.shape[1], " y no cumplen K anonimidad:", ncaract_noanonimas)
    print("f1=", f1)
    print("precision=", accuracy)
    print("falsos positivos=", false_positive)
    print("falsos negativos=", false_negative, "\n\n")

