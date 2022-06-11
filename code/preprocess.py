import utils
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

def preprocess(lowcase = True,
                punctuations = True, numbers = True, 
                whitespaces = True, repeating_chars=True,
                swlang = 'english', swlist = '', train=True):

    if train:
        path = '../../datasets/en/*'
    else:
        path = '../../datasets/en_test/*'

    files = []
    names = []
    for name in glob(path):
        if name.endswith('.xml'):
            files.append(name)
            names.append(name.split('/')[-1][:-4])

    corpusRaw = []
    for file in files:
        xmlfile = ET.parse(file)
        xmlfile = ET.tostring(xmlfile.getroot(), method='text')
        corpusRaw.append(str(xmlfile))

    corpusPreprocess = corpusRaw
    corpusPreprocess = [utils.removingTabs(x) for x in corpusPreprocess]
    if (lowcase):
        print("To lower...")
        corpusPreprocess = [x.lower() for x in corpusPreprocess]
    if (whitespaces):
        print('Whitespaces...')
        corpusPreprocess = [utils.removingMultipleSpaces(x) for x in corpusPreprocess]
    if (punctuations):
        print('Punctuations...')
        corpusPreprocess = [utils.removingPunctuacion(x) for x in corpusPreprocess]
    if numbers:
        print('Numbers...')
        corpusPreprocess = [utils.removingNumbers(x) for x in corpusPreprocess]
    if repeating_chars:
        print('Repeating chars...')
        corpusPreprocess = [utils.removingRepeatingChars(x) for x in corpusPreprocess]

    sw = stopwords.words(swlang)
    print('Removing stopwords...')
    corpusPreprocess = [utils.removingWords(x, sw) for x in corpusPreprocess]

    print('Removing word list...')
    corpusPreprocess = [utils.removingWords(x, swlist) for x in corpusPreprocess]

    print('Removing tweet separator...')
    corpusPreprocess = [re.sub(r'.', '', x, count=1) for x in corpusPreprocess]

    return corpusPreprocess, names

def labelling(corpusDF):
    print("Labelling...")
    labels = pd.read_csv('../../datasets/en/truth.txt', sep=':::', names=['names','label'], dtype={'names':str, 'label':str})
    corpusDF = corpusDF.merge(labels, on='names')
    return corpusDF

def bowBinary(text_tokens, vocabulary):
    sentenceVector = []
    for word in vocabulary:
        if word in text_tokens:
            sentenceVector.append(1)
        else:
            sentenceVector.append(0)
    return sentenceVector

def bowAbsFreq(freqTokens, vocabulary):
    sentenceVector = []
    for word in vocabulary:
        if word in freqTokens.keys():
            sentenceVector.append(freqTokens[word])
        else:
            sentenceVector.append(0)
    return sentenceVector

def generateCorpusDF(corpusPreprocess, names):
    print("Generating DataFrame...")
    tknz = RegexpTokenizer(r'\w+')
    fullText = ' '.join(corpusPreprocess)
    fullTextTknz = tknz.tokenize(fullText)
    freqDict = dict(Counter(fullTextTknz))
    corpusDF = pd.DataFrame(data=corpusPreprocess, columns = ['text'])
    corpusDF['tokens'] = corpusDF['text'].apply(tknz.tokenize)
    fullText = []
    for token_list in corpusDF['tokens']:
        fullText.append(token_list)
    corpusDF['freqTokens'] = corpusDF['tokens'].apply(lambda x: Counter(x))
    corpusDF['names'] = names

    return corpusDF, freqDict

def generateVocabulary(freqDict, n=1000):
    print("Generating Vocabulary...")
    freqDF = pd.DataFrame(freqDict.keys(), columns=['Words'])
    freqDF['Count']= freqDict.values()
    freqDF = freqDF.sort_values('Count', ascending=False).head(n).reset_index(drop=True)
    return freqDF

def generateBowMatrix(bowDF, bow='bowAbsFreq', train=True):
    bowMatrix = pd.DataFrame()
    for row in bowDF.index:
        aux = pd.DataFrame([bowDF[bow][row]])
        bowMatrix = bowMatrix.append(aux)
    bowMatrix = bowMatrix.reset_index(drop=True)
    bowDF = bowDF.reset_index(drop=True)
    if train:
        bowMatrix['label'] = bowDF['label'].apply(lambda x: 0 if x=='NI' else 1)
    return bowMatrix


def generateFinalData(swlist, n=1000):
    corpusPreprocess, names = preprocess(swlist=swlist)
    corpusDF, freqDict = generateCorpusDF(corpusPreprocess, names)
    corpusDF = labelling(corpusDF)
    vocabulary = generateVocabulary(freqDict, n=n)
    word_vocabulary = vocabulary['Words']

    corpusDF['bowAbsFreq'] = corpusDF['freqTokens'].apply(lambda x: bowAbsFreq(x, word_vocabulary))
    bowDF = corpusDF[['label','names','bowAbsFreq']]
    bowMatrix = generateBowMatrix(bowDF)

    return bowMatrix
