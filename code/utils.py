import pandas as pd
import re
import string


def removingTabs(text):
    text = text.replace('\\n', ' ')
    text = text.replace('\\t', ' ')
    return text

def removingMultipleSpaces(text):
    text = re.sub(r"\s+", ' ', text)
    return text

def removingNumbers(text):
    text = re.sub('[0-9]+', '', text)
    return text

def removingPunctuacion(text):
    punctuation_list = string.punctuation
    translator = str.maketrans('', '', punctuation_list)
    text = text.translate(translator)
    return text

def removingRepeatingChars(text):
    text = re.sub(r'(.)1+',r'1', text)
    return text

def removingWords(text, words):
    return " ".join([word for word in str(text).split() if word not in words])

def labelling(corpusDF):
    print("Labelling...")
    labels = pd.read_csv('../datasets/en/truth.txt', sep=':::', names=['names','label'], dtype={'names':str, 'label':str})
    corpusDF = corpusDF.merge(labels, on='names')
    return corpusDF

swlist = ['user', 'hashtag','url']
