import preprocess
import ml_results

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import swlist
import warnings


def generateTfidfDF(swlist=swlist, train=True):
    warnings.filterwarnings("ignore")
    corpusPreprocess, names = preprocess.preprocess(swlist, train=train)
    corpusDF = preprocess.generateCorpusDF(corpusPreprocess, names)[0]
    if train:
        corpusDF = preprocess.labelling(corpusDF)
        tfid_df = corpusDF[['text','label']]
        tfid_df['label'] = tfid_df['label'].apply(lambda x: 0 if x=='NI' else 1)
    else:
        tfid_df = corpusDF[['text']]
        tfid_df['names'] = names
    return tfid_df

if __name__ == '__main__':
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=10000)

    tfid_df = generateTfidfDF()
    X_train, X_test, y_train, y_test = train_test_split(tfid_df['text'], tfid_df['label'], test_size=0.2, random_state=0)
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)

    RFscores = ml_results.resultsRF(X_train, X_test, y_train, y_test, verbose=True)

    SVCscores = ml_results.resultsSVC(X_train, X_test, y_train, y_test, verbose=True)


