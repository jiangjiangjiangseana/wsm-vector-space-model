from pprint import pprint
from Parser import Parser
from textblob import TextBlob as tb
import util
import tfidf
import nltk

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = {}

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents={}, weighting='tf'):
        self.documentVectors={}
        self.parser = Parser()
        self.blobList = []
        self.keyerror_count = 0
        if(len(documents)>0):
            self.build(documents, weighting)

    def build(self,documents, weighting):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(list(documents.values()))
        for key, value in documents.items():
            self.blobList.append(tb(value))
            self.documentVectors[key] = self.makeVector(value, weighting)

        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        # #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1

        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString, weighting):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)        
        documentString = " ".join(wordList)
        blob = tb(documentString)
        ### tf weighting
        for word in wordList:
            if word in self.vectorKeywordIndex.keys():
                if weighting == 'tf':
                    vector[self.vectorKeywordIndex[word]] += 1 / len(wordList)  #Use simple Term Count Model
                    # vector[self.vectorKeywordIndex[word]] = tfidf.tf(word, blob)
                elif weighting == 'tfidf':
                    vector[self.vectorKeywordIndex[word]] = tfidf.tfidf(word, blob, self.blobList)
            else:
                self.keyerror_count+=1
        return vector


    def buildQueryVector(self, termList, weighting):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList), weighting)
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = {}
        for key, value in documentVectors.items():
            rating = util.cosine(self.documentVectors[documentId], value) 
        #ratings.sort(reverse=True)
            ratings[key] = rating
        return ratings


    def search(self,searchList, formula="cosine", weighting="tf"):
        """ search for documents that match based on a list of terms """
        ratings = {}
        queryVector = self.buildQueryVector(searchList, weighting)
        for key, value in self.documentVectors.items():
            if formula == "cosine":
                rating = util.cosine(queryVector, value) 
            elif formula == "euclidean":
                rating = util.euclidean(queryVector, value)
            ratings[key] = rating
        ratings = {k: v for k, v in sorted(ratings.items(), key=lambda item: item[1], reverse=True)}
        return ratings

    def relevence_search(self, searchVector, formula="cosine", weighting='tf'):
        ratings = {}
        for key, value in self.documentVectors.items():
            if formula == "cosine":
                rating = util.cosine(searchVector, value) 
            elif formula == "euclidean":
                rating = util.euclidean(searchVector, value)
            ratings[key] = rating
        ratings = {k: v for k, v in sorted(ratings.items(), key=lambda item: item[1], reverse=True)}
        print("key error count relevence search:", self.keyerror_count)
        return ratings


    def getRelevenceVector(self, wordString):
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        feedbackWord = []
        pos_result = nltk.pos_tag(wordList)
        for word in pos_result:
            if word[1] == 'VB' or 'NN':
                feedbackWord.append(word[0])
        weighting="tfidf"
        feedbackVector = self.makeVector(" ".join(feedbackWord), weighting)

        return feedbackVector



if __name__ == '__main__':
    #test data
    documents = {"N1":"The cat in the hat disabled",
                 "N2":"A cat is a fine pet ponies.",
                 "N3":"Dogs and cats make good pets.",
                 "N4":"I haven't got a hat."}

    vectorSpace = VectorSpace(documents)

    #print(vectorSpace.vectorKeywordIndex)

    # print("document Vectors: ",vectorSpace.documentVectors)

    # print(vectorSpace.related(1))

    print(vectorSpace.search(["cat", "dog"]))

###################################################
