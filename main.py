import os
import argparse
import tfidf
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
    vectorSpace_tfidf = VectorSpace(documents, weighting='tfidf')


    #1-1
    print('')
    print('WSM Project 1: Ranking by Vector Space Models\n')
    print('1-1')
    print('-------------------------------------')
    print('Term Frequency Weighting + Cosine Similarity')
    sorted_ratings_1 = vectorSpace_tf.search(queryList)
    result(dict(list(sorted_ratings_1.items())[:10]))

    #1-2
    print('')
    print('1-2')
    print('-------------------------------------')
    print('Term Frequency Weighting + Euclidean Distance')
    sorted_ratings_2 = vectorSpace_tf.search(queryList, formula="euclidean")
    result(dict(list(sorted_ratings_2.items())[:10]))

    #1-3
    print('')
    print('1-3')
    print('-------------------------------------')
    print('TF-IDF Weighting + Cosine Similarity')
    sorted_ratings_3 = vectorSpace_tfidf.search(queryList, weighting="tfidf")
    result(dict(list(sorted_ratings_3.items())[:10]))

    #1-4
    print('')
    print('1-4')
    print('-------------------------------------')
    print('TF-IDF Weighting + Euclidean Distance')
    sorted_ratings_4 = vectorSpace_tfidf.search(queryList, formula="euclidean", weighting="tfidf")
    result(dict(list(sorted_ratings_4.items())[:10]))