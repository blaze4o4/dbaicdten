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

class ICD10:
    def __init__(self):
        data_file = pkg_resources.resource_filename('dbaicd10.resources', "dba_icd10.csv")
        vocabulary_file = pkg_resources.resource_filename('dbaicd10.resources', "vocab_list.pkl")

        ## setting data and vocabulary
        self.data = pd.read_csv(data_file)
        infile = open(vocabulary_file, 'rb')
        self.vocab_list = pickle.load(infile)
        infile.close()

        self.stop_words = set(stopwords.words('english'))

    # @memoize
    # @staticmethod
    @memoize
    def levenshtein(self, s, t):
        if s == "" or t == "":
            return max(len(s), len(t))

        if s[-1] == t[-1]:
            cost = 0
        else:
            cost = 1

        res = min([self.levenshtein(s[:-1], t) + 1,
                   self.levenshtein(s, t[:-1]) + 1,
                   self.levenshtein(s[:-1], t[:-1]) + cost])
        # print(res)
        return res

    def auto_correct(self, sentence, remove_stop_words=False, vocab=None, threshold=70):
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
                dist = self.levenshtein(vocab_word, word)
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
        applicable_tos = ast.literal_eval(applicable_tos)
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
        clinical_infos = ast.literal_eval(clinical_infos)
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

    def search(self, keyword, auto_correct_keywords=True, show_time_spent=True, return_top_n=10):

        before = datetime.datetime.now()

        keywords = keyword.split()

        keywords = " ".join([d for d in keywords if d not in self.stop_words])
        if auto_correct_keywords:
            keywords = self.auto_correct(keywords).split()
        else:
            keywords = keywords.split()
        print('Searching for: "' + " ".join(keywords) + '"')

        result = self.data.apply(self.search_helper, axis=1, keywords=keywords)

        after = datetime.datetime.now()

        diff = after - before

        print("Search completed in", diff.seconds, "seconds")

        return self.data.loc[result.nlargest(return_top_n, keep='first').index]

