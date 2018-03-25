"""Given a text file containing words, assigns each word a POS tag.

POS (part-of-speech) tags classify words based on their contextual function.
Examples of POS tags include 'NN' and 'VB', which stand for 'noun' and 'verb'
respectively.

Computer programs can perform POS tagging using one of two major methods:
defining sets of rules, or training on pre-tagged data sets. The most
effective taggers combine these methods as needed. This implementation
relies primarily on training data, but has defined a few rules as well.

Usage:
    The program requires two files -- one for training data and one
    for testing.

    Command-line syntax:
        python tagger.py [training-file] [test-file]

    By default, all output is printed to the standard output (STDOUT).
    Using the redirect operator allows the user to specify a file name
    as STDOUT.

    Syntax with redirect:
        python tagger.py [training-file] [test-file] > [output-file]

Example Input:
    Training:
        [ Pierre/NNP Vinken/NNP ]
        ,/,
        [ 61/CD years/NNS ]
        old/JJ ,/, will/MD join/VB
        [ the/DT board/NN ]
        as/IN
        [ a/DT nonexecutive/JJ director/NN Nov./NNP 29/CD ]
        ./.
        [ Mr./NNP Vinken/NNP ]
        is/VBZ

    Test:
        No ,
        [ it ]
        [ was n't Black Monday ]
        .
        But while
        [ the New York Stock Exchange ]
        did n't
        [ fall ]
        apart
        [ Friday ]

Example Output:
    No/DT ,/,
    [ it/PRP ]
    [ was/VBD n't/RB Black/NNP Monday/NNP ]
    ./.
    But/CC while/IN
    [ the/DT New/NNP York/NNP Stock/NNP Exchange/NNP ]
    did/VBD n't/RB
    [ fall/VB ]
    apart/NN
    [ Friday/NNP ]

Algorithm:
    - program starts in main()
    - main() calls generate_model()

    - generate_model():
        - initializes variables for storing model
        - reads training data
        - uses data to:
            - create tag-word conditional freq. distribution
            - create tag bigram freq. distribution

        - reads test data to generate output string
        - assigns tags to each word in test:
            - if the word is unknown, tags based on rules
            - if the word in known, tags using HMM

        - prints final output string to STDOUT

    - control is passed back to main()
    - program ends

:author name: Srijan Yenumula, Rav Singh
:class: AIT-590, IT-499-002P
:date: 19-MAR-2018
"""

import sys
from collections import defaultdict

from nltk import ConditionalFreqDist, bigrams, str2tuple


def generate_model():
    """Creates the statistical model for POS tagging"""
    brackets = ['[', ']']

    # conditional freq of each word given a tag
    tag_cfdist = ConditionalFreqDist()

    # conditional freq of each tag given another tag
    tag_bigrams = None

    # map of words to tags
    tag_dict = defaultdict(lambda: [])

    # final result, will be printed to file
    output = ''

    # model creation from training data
    with open(sys.argv[1], 'r') as file_:
        # word-tag pairs
        pairs = tuple(
            str2tuple(token.split('|')[0])
            for token in file_.read().split()
            if token not in brackets
        )

        for word, tag in pairs:
            tag_dict[word].append(tag)
            tag_cfdist[tag][word] += 1

        tag_bigrams = ConditionalFreqDist(
            bigrams(
                pair[1]
                for pair in pairs
            )
        )

    test_data = []
    with open(sys.argv[2], 'r') as file_:
        test_data = file_.read().splitlines()

    # guess plausible tag for first word
    prev_tag = 'VB'
    for line in test_data:
        for word in line.split():
            if word == brackets[0]:
                output += '[ '
            elif word == brackets[1]:
                output += ']'
            else:
                # assign 'NN' (noun) to unknown words
                if not tag_dict[word]:
                    if word[-1] == 's':
                        prev_tag = 'NNS'
                    elif word[-2:] == 'ed':
                        prev_tag = 'VBN'
                    elif word[0].isupper():
                        prev_tag = 'NNP'
                    elif word[-4:] == 'able' or '-' in word:
                        prev_tag = 'JJ'
                    else:
                        prev_tag = 'NN'

                    output += f'{word}/{prev_tag} '
                    continue

                argmax = 0
                best_tag = ''

                # find the most probable tag
                # formula = P(word|tag) * P(tag|prev_tag)
                for tag in tag_dict[word]:
                    prob = \
                        tag_cfdist[tag].freq(word) \
                        * tag_bigrams[prev_tag].freq(tag)
                    if prob >= argmax:
                        argmax = prob
                        best_tag = tag

                output += f'{word}/{best_tag} '
                prev_tag = best_tag

        # add newline char after each line
        output += '\n'

    print(output)


def main():
    """Starts model generation"""
    generate_model()


if __name__ == '__main__':
    main()
