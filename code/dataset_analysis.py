import preprocess
import numpy as np
import matplotlib.pyplot as plt

def generate_plot(vocabulary):
    x = np.linspace(0, len(vocabulary), len(vocabulary))
    plt.plot(x,vocabulary['Count'])
    plt.ylabel('Word Count')
    plt.xlabel('Word number')
    plt.title('All words')
    plt.show()
    x = np.linspace(0, len(vocabulary[:200]), len(vocabulary[:200]))
    plt.plot(x,vocabulary['Count'][:200])
    plt.ylabel('Word Count')
    plt.xlabel('Word number')
    plt.title('200 words')
    plt.show()

def generate_histogram(vocabulary):
    plt.bar(vocabulary['Words'].head(10), vocabulary['Count'].head(10))
    plt.ylabel('Count')
    plt.title('Vocabulary histogram')
    plt.show()

if __name__ == '__main__':
    swlist = ['user', 'hashtag','url']
    corpusPreprocess, names = preprocess.preprocess(swlist=swlist)
    corpus_df_train, freq_dict_train = preprocess.generateCorpusDF(corpusPreprocess, names)
    vocab = preprocess.generateVocabulary(freq_dict_train)

    generate_histogram(vocab)
    generate_plot(vocab)




