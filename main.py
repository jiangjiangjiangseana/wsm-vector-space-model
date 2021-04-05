import os
import gc
import argparse
import tfidf
import numpy as np
from textblob import TextBlob as tb
from VectorSpace import VectorSpace


def get_query():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--query', dest='query', type=str,
        help='Get query input')
    
    return parser.parse_args()

def load_files_in_dir():
    path = os.getcwd() + '\\EnglishNews'
    all_files = os.listdir(path)
    
    return all_files

def read_documents(file_list):
    documents = {}
    news_path = os.getcwd() + '\\EnglishNews'
    for file in file_list:
        file_path = news_path +'\\' + file
        with open(file_path, 'r', encoding = 'utf-8') as f:
            documents[file[:-4]] = f.read()
    
    return documents

def result(sorted_dic_10):
    print('\nNews ID                Score')
    print('------------           ------------')
    for key, value in sorted_dic_10.items():
        print(key, '           ', round(value, 7))

if __name__ == '__main__':
    
    query = get_query().query
    queryList = list(query.split(' '))
    print("my query is ", queryList)
    engNewsList = load_files_in_dir()
    documents = read_documents(engNewsList)
    vectorSpace_tf = VectorSpace(documents)
    


    #1-1
    print('')
    print('WSM Project 1: Ranking by Vector Space Models\n')
    print('1-1')
    print('-------------------------------------')
    print('Term Frequency Weighting + Cosine Similarity')
    sorted_ratings_1 = vectorSpace_tf.search(queryList)
    top_10_tf_cos = dict(list(sorted_ratings_1.items())[:10])
    result(top_10_tf_cos)

    #1-2
    print('')
    print('1-2')
    print('-------------------------------------')
    print('Term Frequency Weighting + Euclidean Distance')
    sorted_ratings_2 = vectorSpace_tf.search(queryList, formula="euclidean")
    top_10_tf_dis = dict(list(sorted_ratings_2.items())[:10])
    result(top_10_tf_dis)


   
    vectorSpace_tfidf = VectorSpace(documents, weighting='tfidf')

    #1-3
    print('')
    print('1-3')
    print('-------------------------------------')
    print('TF-IDF Weighting + Cosine Similarity')
    sorted_ratings_3 = vectorSpace_tfidf.search(queryList, weighting="tfidf")
    top_10_tfidf_cos = dict(list(sorted_ratings_3.items())[:10])
    result(top_10_tfidf_cos)

    #1-4
    print('')
    print('1-4')
    print('-------------------------------------')
    print('TF-IDF Weighting + Euclidean Distance')
    sorted_ratings_4 = vectorSpace_tfidf.search(queryList, formula="euclidean", weighting="tfidf")
    top_10_tfidf_dis = dict(list(sorted_ratings_4.items())[:10])
    result(top_10_tfidf_dis)


    #2
    print('')
    print('2')
    print('-------------------------------------')
    print('Relevence Feedback - TF-IDF + Cosine Similarity')
    top_tfidf_cos_idx = list(sorted_ratings_3.items())[0][0]
    top_doc = documents[top_tfidf_cos_idx]
    feedbackVector = vectorSpace_tfidf.getRelevenceVector(top_doc)
    queryVector = vectorSpace_tfidf.buildQueryVector(queryList, weighting="tfidf")
    rf_query_vector = list(1 * np.array(queryVector) + 0.5 * np.array(feedbackVector))
    sorted_ratings_5 = vectorSpace_tfidf.relevence_search(rf_query_vector, weighting='tfidf')
    relevence_ratings = dict(list(sorted_ratings_5.items())[:10])
    result(relevence_ratings)