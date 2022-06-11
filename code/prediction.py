import warnings

from tfidf_evaluation import generateTfidfDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from lxml import etree

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    train_df = generateTfidfDF()
    X_train, y_train = train_df['text'], train_df['label']
    vectoriser.fit(X_train)

    test_df = generateTfidfDF(train=False)

    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(test_df['text'])
    authors = test_df['names']

    model = RandomForestClassifier(max_depth=50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_label = []
    for label in y_pred:
        if label == 0:
            y_label.append('NI')
        else:
            y_label.append('I')


    for i in range(len(authors)):
        root = etree.Element("author", id=authors[i], lang='en', type=y_label[i])
        tree = etree.ElementTree(root)
        tree.write("../../datasets/prediction/"+authors[i]+".xml")



