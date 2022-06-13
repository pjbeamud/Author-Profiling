# Author profiling / ES

En este trabajo hemos tratado de crear un modelo de machine learning capaz de predecir si un perfil de twitter es irónico o no utilizando un dataset facilitado por:
https://pan.webis.de/clef22/pan22-web/author-profiling.html#evaluation

## Instalación

Queda adjuntado un achivo: "requirements.txt" donde se encuentran las librerías necesarias.

## Descripción de los scripts

- utils.py: Funciones para la preparación del dataset
- preprocess.py: Preprocesamiento del dataset, generación del vocabulario y formación de la matriz de características.
- dataset_analysis.py: pequeño análisis del corpus
- ml_results.py: los modelos de machine learning que usaremos
- vocabulary_evaluation.py: evaluación de los modelos de machine learning para diferentes tamaños de vocabulario
- cross_validation.py: evaluación de los resultados mediante cross validation.
- tfidf_evaluation.py: evaluación del modelo de TF/IDF.
- prediction.py: generación del resultado mediante TF/IDF.

# Author profiling / EN

In this project we present a model to evaluate using machine learning if an author is ironic or not. Dataset source:
https://pan.webis.de/clef22/pan22-web/author-profiling.html#evaluation

## Install

You can find a "requirements.txt" file with all dependencies.

## Script description

- utils.py: useful functions to preprocess corpus
- preprocess.py: preprocess of the dataset, generate vocabulary and creation of feature matrix.
- dataset_analysis.py: simple analysis of dataset
- ml_results.py: all machine learning models we use
- vocabulary_evaluation.py: evaluation of our machine learning models using different vocabularies
- cross_validation.py: evaluation using cross validation.
- tfidf_evaluation.py: evaluation of TF/IDF model.
- prediction.py: generation of results with TF/IDF.











