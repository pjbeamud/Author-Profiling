from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from sklearn.svm import LinearSVC
from keras.layers import Dense
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

def resultsKN(X_train, X_test, y_train, y_test, verbose=False, pred = False):
    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    if verbose:
        print("--------------------")
        print("Classification Report for KN")
        print(classification_report(y_test, y_pred))
        print("--------------------")
        print("Accuracy for KN:", accuracy)
    if pred:
        return y_pred
    return accuracy, f1, recall, precision

def resultsSVC(X_train, X_test, y_train, y_test, verbose=False, pred = False):
    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    if verbose:
        print("--------------------")
        print("Classification Report for SVC")
        print(classification_report(y_test, y_pred))
        print("--------------------")
        print("Accuracy for SVC:", accuracy)
    if pred:
        return y_pred
    return accuracy, f1, recall, precision

def resultsNB(X_train, X_test, y_train, y_test, verbose=False, pred = False):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    if verbose:
        print("--------------------")
        print("Classification Report for NB")
        print(classification_report(y_test, y_pred))
        print("--------------------")
        print("Accuracy for NB:", accuracy)
    if pred:
        return y_pred
    return accuracy, f1, recall, precision

def resultsRF(X_train, X_test, y_train, y_test, verbose=False, pred = False):
    model = RandomForestClassifier(max_depth=50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    if verbose:
        print("--------------------")
        print("Classification Report for RF")
        print(classification_report(y_test, y_pred))
        print("--------------------")
        print("Accuracy for RF:", accuracy)
    if pred:
        return y_pred
    return accuracy, f1, recall, precision

def resultsNN(X_train, X_test, y_train, y_test, vocab_len, pred = False):
    model = Sequential()
    model.add(Dense(12, input_dim=vocab_len, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    evaluation = model.evaluate(X_test, y_test)
    accuracy = evaluation[1]
    precision = evaluation[2]
    recall = evaluation[3]
    f1 = 2*(precision * recall)/(precision + recall)
    y_pred = model.predict(X_test)
    if pred:
        return y_pred
    return accuracy, f1, recall, precision
    
def allModelsResults(X_train, X_test, y_train, y_test, vocab_len):
    resultSVC = resultsSVC(X_train, X_test, y_train, y_test)
    resultNB = resultsNB(X_train, X_test, y_train, y_test)
    resultRF = resultsRF(X_train, X_test, y_train, y_test)
    resultKN = resultsKN(X_train, X_test, y_train, y_test)
    resultNN = resultsNN(X_train, X_test, y_train, y_test, vocab_len)
    return resultSVC, resultNB, resultRF, resultKN, resultNN
