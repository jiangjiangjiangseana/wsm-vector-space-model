import os
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
    