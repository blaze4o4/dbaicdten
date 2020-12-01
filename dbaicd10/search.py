import re

import pandas as pd
import numpy as np

import ast
import pickle

import datetime

from nltk.corpus import stopwords
import pkg_resources
# from pkg_resources import resource_string, resource_listdir

def memoize(func):
    memory = {}
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in memory:
            memory[key] = func(*args, **kwargs)
        return memory[key]

    return memoizer

@memoize
def levenshtein( s, t):
    if s == "" or t == "":
        return max(len(s), len(t))

    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1

    res = min([levenshtein(s[:-1], t) + 1,
               levenshtein(s, t[:-1]) + 1,
               levenshtein(s[:-1], t[:-1]) + cost])
    # print(res)
    return res

class ICD10:
    def __init__(self):
        data_file = pkg_resources.resource_filename('dbaicd10.resources', "dba_icd10.csv")
        vocabulary_file = pkg_resources.resource_filename('dbaicd10.resources', "vocab_list.pkl")

        ## setting data and vocabulary
        self.data = pd.read_csv(data_file)
        self.data['Approximate Synonyms'] = self.data['Approximate Synonyms']\
                                                .apply(lambda x: ast.literal_eval(x))

        self.data['Applicable To'] = self.data['Applicable To'] \
            .apply(lambda x: ast.literal_eval(x))

        self.data['Clinical Info'] = self.data['Clinical Info'] \
            .apply(lambda x: ast.literal_eval(x))

        infile = open(vocabulary_file, 'rb')
        self.vocab_list = pickle.load(infile)
        infile.close()

        self.stop_words = set(stopwords.words('english'))

    # @memoize
    # @staticmethod


    def auto_correct(self, sentence, remove_stop_words=False, vocab=None, threshold=70):
        '''
        Auto corrects a sentence from a vocabulary based on ICD10 dataset
        :param sentence: (String) text that needs to be autocorrects
        :param remove_stop_words: (boolean) whether to remove stopwords from sentence
        :param vocab: (list of string) If need to provide a custom vocabulary
        :param threshold: ( Integer: Default=70) Corrects the word if it matches atleast threshold percent from any word from vocabulary
        :return: (String) autocorrected sentence
        '''
        ## Preprocessing
        sentence = sentence.lower()
        ### Make alphanumeric
        sentence = re.sub(r'\W+', ' ', sentence)
        ## remove double spaces
        sentence = re.sub(' +', ' ', sentence)

        allowed_error = 1 - (threshold / 100)

        if vocab is None:
            vocab = self.vocab_list

        words = sentence.split()

        final_sent = ''

        for word in words:
            ## for each wors we will find in vocabulary, the vocab_word with least distance
            distance = 9999
            best_match = None
            for vocab_word in vocab:
                dist = levenshtein(vocab_word, word)
                if dist < distance:
                    distance = dist
                    best_match = vocab_word
            if distance < allowed_error * len(word):
                final_sent = final_sent + " " + best_match
            else:
                final_sent = final_sent + " " + word
        return final_sent.strip()

    def search_helper(self, row, keywords):
        ## first search in name
        #     print( keywords)


        # Step 1: Score of Name ( score = how many words match )
        name = row['name'].lower().split()
        #     print(name)
        name_score = 0
        for keyword in keywords:
            if keyword.lower().strip() in name:
                name_score += 1

                #     print(name_score)

        ## Step 2: Socre of approximate synonyms
        ## now search in approximate synonyms
        synonyms = row['Approximate Synonyms']
        #     synonyms = ast.literal_eval(synonyms)
        #     print(synonyms)
        syn_scores = [0] * len(synonyms)

        # there are multiple synonyms for each row,
        # so we find score for each of them
        for i, synonym in enumerate(synonyms):
            synonym = synonym.lower().split()
            for keyword in keywords:
                if keyword.lower() in synonym:
                    syn_scores[i] += 1
        # score of synonym is max of score of each synonym


        synonym_score = np.max(syn_scores)

        ## Step 3: Score of applicable two
        ## now search in Applicable To
        applicable_tos = row['Applicable To']
        # applicable_tos = ast.literal_eval(applicable_tos)
        # print(applibable_tos[0])
        # for dk in
        #     synonyms = ast.literal_eval(synonyms)
        #     print(synonyms)
        applicable_scores = [0] * len(applicable_tos)

        ## there are multiple applicable to for each row
        # so we find score for each of them

        for i, applicable in enumerate(applicable_tos):
            # if applicable == 'Tennis elbow':
            #   print('Tennis elbow found')
            # print(applicable)
            applicable = applicable.lower().split()
            for keyword in keywords:
                if keyword.lower() in applicable:
                    applicable_scores[i] += 1
        # score of synonym is max of score of each synonym
        applicable_score = np.max(applicable_scores)

        ## STEP 4: Score of Clinical Info
        ## now search in Applicable To
        clinical_infos = row['Clinical Info']
        # clinical_infos = ast.literal_eval(clinical_infos)
        #     print(synonyms)
        clinical_scores = [0] * len(clinical_infos)

        ## there are multiple applicable to for each row
        # so we find score for each of them


        for i, clinical in enumerate(clinical_infos):
            clinical = clinical.lower().split()
            for keyword in keywords:
                if keyword.lower() in clinical:
                    clinical_scores[i] += 1
        # score of synonym is max of score of each synonym
        clinical_score = np.max(clinical_scores)

        #     print(syn_score)

        # we return the score which is better name or synonym

        # print([name_score, synonym_score, applicable_score, clinical_score])

        return np.max([name_score, synonym_score, applicable_score, clinical_score])

    def search_helper2(self, row, keywords):

        INCREMENT_SCORE_BY = 1

        ## first search in name
        #     print( keywords)
        ## just makeone big string of all columns, and see how many of keywords we can find
        all_cols = ''
        all_cols += row['name'].lower()

        all_cols += " ".join(row['Approximate Synonyms'])

        # score of clinical info should be less than others
        clinical_info = " ".join(row['Clinical Info'])

        all_cols += " ".join(row['Applicable To'])

        # lower
        all_cols = all_cols.strip().lower()
        all_cols = re.sub(r'\W+', ' ', all_cols)
        ## remove double spaces
        all_cols = re.sub(' +', ' ', all_cols)

        # all_words = all_cols.split()

        score = 0
        ## searcg for keywords
        for keyword in keywords:
            ## SOME OPTIMIZATIONS: Here we are calculating few thinks which we will require multiple times
            SPACE_IN_KEYWORD = ' ' in keyword
            KEYWORD_SPLIT = keyword.split()

            ## if we find exact keyword ( example: "muscle fatigue" ) then score is
            ## increased 1
            if keyword in all_cols:
                ## if keyword is of multple words, and it matches means it should increase score more
                if SPACE_IN_KEYWORD:
                    score += 1.23 * INCREMENT_SCORE_BY
                else:
                    score += INCREMENT_SCORE_BY
            ## else we find if keyword can be further divided into smaller keywords
            elif SPACE_IN_KEYWORD:
                for temp_keyword in KEYWORD_SPLIT:
                    if temp_keyword in all_cols:
                        score += 0.23 * INCREMENT_SCORE_BY

            ## if found in clinical info, increase the score, but less
            if keyword in clinical_info:
                score += INCREMENT_SCORE_BY * 0.6
            elif SPACE_IN_KEYWORD:
                for temp_keyword in KEYWORD_SPLIT:
                    if temp_keyword in clinical_info:
                        score += 0.23 * INCREMENT_SCORE_BY * 0.6

                        ## extra scores
            ## if keyword is in name only then we give it extra score
            if keyword in row['name'].lower():
                score += INCREMENT_SCORE_BY * 0.23
            elif SPACE_IN_KEYWORD:
                for temp_keyword in KEYWORD_SPLIT:
                    if temp_keyword in row['name']:
                        score += 0.1 * INCREMENT_SCORE_BY

        return score

    def search(self, keyword, auto_correct_keywords=True, show_time_spent=True, return_top_n=10,
               split_with_stopwords=True):
        '''
        Search in ICD10 dataset for the provided keywords. It performs a simple match word search.
        :param keyword: (String) keywords or sentence to search for. Keywords seperated by space
        :param auto_correct_keywords: (Boolean: default=True) Keep it true for spell check of the given keywords
        :param show_time_spent: (Boolean: default=True) Display time utilized for search
        :param return_top_n: (integer: default:10) Returns the number of top results. Is set to 10 returns top 10 results
        :param split_with_stopwords: (Boolean: default=True) Keep it true if you want to split the search query from stopwords instead of space. Refer example below for more info
        :return: Returns a pandas dataframe with top matches
        use case:
        search("Tennis elbow")
        search("tennis elbo", auto_correct_keywords=True, return_top_n=5)

        Example of split_with_stopwords:
        There might be cases where you want to keep two words together, instance "Right Hand"
        So here we split the query from stopwords instead of spaces. Thus,
        "Fracture in right hand" becomes => ["fracture", "right hand"] instead of "fracture", "right", hand"]
        Note that "in" was the stopword and query got splitted from "in"
        '''
        before = datetime.datetime.now()
        keyword = keyword.lower()

        if auto_correct_keywords:
            keyword = self.auto_correct(keyword)

        if split_with_stopwords:
            for stopword in self.stop_words:
                if stopword in keyword:
                    keyword = keyword.replace(' ' + stopword + ' ', '#')
            keywords = keyword.split('#')
        else:
            keywords = keyword.split()
            keywords = " ".join([d for d in keywords if d not in self.stop_words])
            keywords = keywords.split()
        print('Searching for: "' + " ".join(keywords) + '"')

        result = self.data.apply(self.search_helper2, axis=1, keywords=keywords)

        after = datetime.datetime.now()

        diff = after - before

        if show_time_spent:
            print("Search completed in", diff.seconds, "seconds")

        return self.data.loc[result.nlargest(return_top_n, keep='first').index]
