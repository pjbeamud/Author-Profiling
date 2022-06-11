import preprocess
import ml_results
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from utils import swlist

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    corpusPreprocess, names = preprocess.preprocess(swlist=swlist)
    corpusDF, freqDict = preprocess.generateCorpusDF(corpusPreprocess, names)
    corpusDF = preprocess.labelling(corpusDF)

    vocabulary_50 = preprocess.generateVocabulary(freqDict, 50)['Words']
    vocabulary_100 = preprocess.generateVocabulary(freqDict, 100)['Words']
    vocabulary_500 = preprocess.generateVocabulary(freqDict, 500)['Words']
    vocabulary_1000 = preprocess.generateVocabulary(freqDict, 1000)['Words']
    vocabulary_2000 = preprocess.generateVocabulary(freqDict, 2000)['Words']

    corpusDF['bowAbsFreq_50'] = corpusDF['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary_50))
    corpusDF['bowAbsFreq_100'] = corpusDF['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary_100))
    corpusDF['bowAbsFreq_500'] = corpusDF['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary_500))
    corpusDF['bowAbsFreq_1000'] = corpusDF['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary_1000))
    corpusDF['bowAbsFreq_2000'] = corpusDF['freqTokens'].apply(lambda x: preprocess.bowAbsFreq(x, vocabulary_2000))

    bowDF_50 = corpusDF[['label','names','bowAbsFreq_50']]
    bowMatrix_train_50 = preprocess.generateBowMatrix(bowDF_50, bow='bowAbsFreq_50')
    bowDf_100 = corpusDF[['label','names','bowAbsFreq_100']]
    bowMatrix_train_100 = preprocess.generateBowMatrix(bowDf_100, bow='bowAbsFreq_100')
    bowDF_500 = corpusDF[['label','names','bowAbsFreq_500']]
    bowMatrix_train_500 = preprocess.generateBowMatrix(bowDF_500, bow='bowAbsFreq_500')
    bowDf_1000 = corpusDF[['label','names','bowAbsFreq_1000']]
    bowMatrix_train_1000 = preprocess.generateBowMatrix(bowDf_1000, bow='bowAbsFreq_1000')
    bowDf_2000 = corpusDF[['label','names','bowAbsFreq_2000']]
    bowMatrix_train_2000 = preprocess.generateBowMatrix(bowDf_2000, bow='bowAbsFreq_2000')

    X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(bowMatrix_train_50.loc[:, bowMatrix_train_50.columns != "label"], bowMatrix_train_50['label'], test_size=0.2, random_state=0)
    X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(bowMatrix_train_100.loc[:, bowMatrix_train_100.columns != "label"], bowMatrix_train_100['label'], test_size=0.2, random_state=0)
    X_train_500, X_test_500, y_train_500, y_test_500 = train_test_split(bowMatrix_train_500.loc[:, bowMatrix_train_500.columns != "label"], bowMatrix_train_500['label'], test_size=0.2, random_state=0)
    X_train_1000, X_test_1000, y_train_1000, y_test_1000 = train_test_split(bowMatrix_train_1000.loc[:, bowMatrix_train_1000.columns != "label"], bowMatrix_train_1000['label'], test_size=0.2, random_state=0)
    X_train_2000, X_test_2000, y_train_2000, y_test_2000 = train_test_split(bowMatrix_train_2000.loc[:, bowMatrix_train_2000.columns != "label"], bowMatrix_train_2000['label'], test_size=0.2, random_state=0)

    results_50 = ml_results.allModelsResults(X_train_50, X_test_50, y_train_50, y_test_50, 50)
    results_100 = ml_results.allModelsResults(X_train_100, X_test_100, y_train_100, y_test_100, 100)
    results_500 = ml_results.allModelsResults(X_train_500, X_test_500, y_train_500, y_test_500, 500)
    results_1000 = ml_results.allModelsResults(X_train_1000, X_test_1000, y_train_1000, y_test_1000, 1000)
    results_2000 = ml_results.allModelsResults(X_train_2000, X_test_2000, y_train_2000, y_test_2000, 2000)

    accuracy50 = [results_50[0][0],results_50[1][0],results_50[2][0],results_50[3][0],results_50[4][0]]
    accuracy100 = [results_100[0][0],results_100[1][0],results_100[2][0],results_100[3][0],results_100[4][0]]
    accuracy500 = [results_500[0][0],results_500[1][0],results_500[2][0],results_500[3][0],results_500[4][0]]
    accuracy1000 = [results_1000[0][0],results_1000[1][0],results_1000[2][0],results_1000[3][0],results_1000[4][0]]
    accuracy2000 = [results_2000[0][0],results_2000[1][0],results_2000[2][0],results_2000[3][0],results_2000[4][0]]

    x_axis = ['SVC','NB','RF','KN','NN']

    plt.figure()
    plt.plot(x_axis, accuracy50, label='50 words')
    plt.plot(x_axis, accuracy100, label='100 words')
    plt.plot(x_axis, accuracy500, label='500 words')
    plt.plot(x_axis, accuracy1000, label='1000 words')
    plt.plot(x_axis, accuracy2000, label='2000 words')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Results')
    plt.legend()
    plt.show()
