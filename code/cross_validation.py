import ml_results
import preprocess
import warnings

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import KFold
from collections import Counter
from utils import swlist

def generateVocabularyKFOLD(df_train, n=1000):
    text = " ".join(df_train['text'])
    tknz = RegexpTokenizer(r'\w+')
    text_tokenized = tknz.tokenize(text)
    freqDict = dict(Counter(text_tokenized))
    vocabulary = preprocess.generateVocabulary(freqDict, n)['Words']
    return vocabulary

def meanValues(matrix):
    cols = ['SVC', 'NB', 'RF', 'KN', 'NN']
    df = pd.DataFrame(matrix, columns=cols)

    svc_mean = df['SVC'].mean()
    nb_mean = df['NB'].mean()
    rf_mean = df['RF'].mean()
    kn_mean = df['KN'].mean()
    nn_mean = df['NN'].mean()

    return [svc_mean, nb_mean, rf_mean, kn_mean, nn_mean]


def crossValidation(corpusDF, n_splits=10):
    warnings.filterwarnings("ignore")
    kfold = KFold(n_splits=10)
    kfold.get_n_splits(corpusDF)
    accuracyVector = []
    precisionVector = []
    f1Vector = []
    recallVector = []
    for train_index, test_index in kfold.split(corpusDF):
        df_train = corpusDF.loc[train_index]
        df_test = corpusDF.loc[test_index]

        vocabulary = generateVocabularyKFOLD(df_train, n=500)

        df_train['bowAbsFreq'] = df_train['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary))
        bow_df_train = df_train[['label','names','bowAbsFreq']]
        bow_matrix_train = preprocess.generateBowMatrix(bow_df_train)

        df_test['bowAbsFreq'] = df_test['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary))
        bow_df_test = df_test[['label','names','bowAbsFreq']]
        bow_matrix_test = preprocess.generateBowMatrix(bow_df_test)

        X_train = bow_matrix_train.loc[:, bow_matrix_train.columns != "label"]
        X_test = bow_matrix_test.loc[:, bow_matrix_test.columns != "label"]
        y_train = bow_matrix_train["label"]
        y_test = bow_matrix_test["label"]

        resultSVC, resultNB, resultRF, resultKN, resultNN = ml_results.allModelsResults(X_train, X_test, y_train, y_test, 500)
        accuracyVector.append([resultSVC[0], resultNB[0], resultRF[0], resultKN[0], resultNN[0]])
        f1Vector.append([resultSVC[1], resultNB[1], resultRF[1], resultKN[1], resultNN[1]])
        recallVector.append([resultSVC[2], resultNB[2], resultRF[2], resultKN[2], resultNN[2]])
        precisionVector.append([resultSVC[3], resultNB[3], resultRF[3], resultKN[3], resultNN[3]])

    return accuracyVector, f1Vector, recallVector, precisionVector



if __name__ == '__main__':
    corpusPreprocess, names = preprocess.preprocess(swlist=swlist)
    corpusDF, freq_dict_train = preprocess.generateCorpusDF(corpusPreprocess, names)
    corpusDF = preprocess.labelling(corpusDF)

    accuracyVector, f1Vector, recallVector, precisionVector = crossValidation(corpusDF)

    mean_accuracy = meanValues(accuracyVector)
    mean_f1 = meanValues(f1Vector)
    mean_recall = meanValues(recallVector)
    mean_precision = meanValues(precisionVector)

    aux_list = [mean_accuracy, mean_f1, mean_recall, mean_precision]

    results_df = pd.DataFrame(aux_list, columns=['SVC', 'NB', 'RF', 'KN', 'NN'], index=['accuracy','f1','recall','precision'])
    print(results_df.to_string())
