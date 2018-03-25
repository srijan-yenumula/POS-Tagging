'''
   """Scores output from tagger.py by comparing against a key"""
    
    This program will compare the "pos-test-with-tags.txt" (output of tagger.py) with a given
    key "pos-test-key.txt" file and calculates the accuracy and provide the
    confusion matrix for the same.
    
    Libraries Used: nltk, pandas, sys, scikit learn, scipy
    
    Usage:
    The program requires two files --
    1. the output generated file from tagger.py which is pos-test-with-tags.txt in this case
    2. the given key which is pos-test-key.txt
    
    The program should be run from command prompt/ terminal, once the path of the python file is specified
    the below line should be typed:
    
    python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-tagging-report.txt
    
    once the above command is run, a text file pos-tagging-report.txt is created in the
    directory location. This file will contain accuracy and confusion matrix computed for
    the above tag comparisons.
    
    
    Algorithm:
    
    Step 1: Program starts in main()
    
    Step 2: Read the output file and create a data frame with two columns Word1,Tag1
    
    Step 3: Read the POS tagged Key file and create a data frame with two columns Word2, tag2
    
    Step 4: Match corresponding words from the model output to the test key and increment a counter each time when there is a match
    
    Step 5: Find the accuracy of the model by dividing the count of matched words with the len of the model out file.
    
    Step 6: Create a confusion matrix by comparing the tagged  key  POS file tokens  with the model output tagged POS tokens
    
    Step 7: END
    
:author name: Srijan Yenumula, Rav Singh
:class: AIT-590, IT-499-002P
:date: 19-MAR-2018
    
'''

import sys
import nltk
import pandas as pd
import scipy
from sklearn.metrics import confusion_matrix


def main():
    """Program entry point"""

    actual_result_file = sys.argv[1]
    groundtruth_file = sys.argv[2]

    with open(actual_result_file) as f:
        actual_result = f.read()

    actual_tagged_tokens = [
        nltk.tag.str2tuple(t)
        for t in actual_result.split()
        if t.strip() not in ['[', ']']
    ]
    df_actual = pd.DataFrame(actual_tagged_tokens, columns=['word1', 'tag1'])

    with open(groundtruth_file) as f:
        groundtruth = f.read()

    groundtruth_tagged_tokens = [
        nltk.tag.str2tuple(t)
        for t in groundtruth.split()
        if t.strip() not in ['[', ']']
    ]
    df_groundtruth = pd.DataFrame(
        groundtruth_tagged_tokens, columns=['word2', 'tag2'])

    df = pd.concat([df_actual, df_groundtruth], axis=1)
    df['match'] = df['tag1'] == df['tag2']
    tot = df['match'].count()
    correct = df['match'].value_counts()[True]
    # error = df['match'].value_counts()[False]

    tag_key_list = [x[1] for x in groundtruth_tagged_tokens]
    model_tag_list = [x[1] for x in actual_tagged_tokens]
    count = 0

    for word in tag_key_list:
        pos = tag_key_list.index(word)
        if (word == model_tag_list[pos]):
            count = count+1

    accuracy = (correct/tot)*100

    outvalue = "Accuracy is {}".format(accuracy)

    pos_key = set(tag_key_list)
    list(pos_key)
    model_key = set(model_tag_list)
    list(model_key)
    # creating confusion matrix
    df1 = pd.Series((v for v in tag_key_list))
    df2 = pd.Series((v for v in model_tag_list))

    df_confmatrix = pd.crosstab(df1, df2)

    print(
        outvalue,
        '\n',
        'Confusion Matrix: ',
        str(df_confmatrix),
        sep='\n',
    )


if __name__ == '__main__':
    main()
