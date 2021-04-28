import re
import spacy
import conll
from spacy.tokens import Doc

from nltk.metrics import *

# utils
import utils

# global vars
nlp = spacy.load('en_core_web_sm')

# whitespace tokenizer


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

data_folder_path = './data'
train_path = data_folder_path + '/train.txt'
test_path = data_folder_path + '/test.txt'

# load data from file and preprocessing


def load_conll_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    sentences = data.split('\n\n')
    OUTPUT_DATA = []
    entities = []
    for sent in sentences:
        tokens = sent.split('\n')
        sentence = []
        ent_sentence_spacy = []
        ents = []

        if tokens[0] != '-DOCSTART- -X- -X- O' and tokens[0] != '':
            for x in tokens:
                x_split = x.split()
                # if not short length
                if len(x) > 0 and len(x_split) >= 3:
                    word = x_split[0]
                    word = word.strip()

                    if len(word) > 0:
                        sentence.append(word)
                        try:
                            ent = x_split[-1]
                        except IndexError:
                            print('Index Error: ', x_split)
                        ents.append((word, ent))
                # else:
                    #print('Short length x: ', x, ' . Removed.')

            processed_sentence = ' '.join(sentence)  # .lower()
            OUTPUT_DATA.append((processed_sentence, {'entities': ents}))

    print('Done getting data !')
    print('There are %d sentences.' % (len(sentences)))
    return OUTPUT_DATA


def convert_spacy_entity_to_conll(iob, type):
    mapper = {
        'PERSON': 'PER',
        'NORP': 'MISC',
        'FAC': 'MISC',
        'ORG': 'ORG',
        'GPE': 'LOC',
        'LOC': 'LOC',
        'PRODUCT': 'MISC',
        'EVENT': 'MISC',
        'WORK_OF_ART': 'MISC',
        'LAW': 'MISC',
        'LANGUAGE': 'MISC',
        'DATE': 'MISC',
        'TIME': 'MISC',
        'PERCENT': 'MISC',
        'MONEY': 'MISC',
        'QUANTITY': 'MISC',
        'ORDINAL': 'MISC',
        'CARDINAL':  'MISC'
    }

    if not type in mapper:
        return 'O'
    return iob + '-' + mapper[type]


# def get_accuracy(data):
#     result = {}
#     iob_tags = ['B', 'I']
#     type_tags = ['PER', 'ORG', 'LOC', 'MISC']

#     # init result object
#     for i in iob_tags:
#         for j in type_tags:
#             key = i + '-' + j
#             result[key] = {'correct': 0, 'total': 0}
#     # add 'O' to result
#     result['O'] = {'correct': 0, 'total': 0}

#     index = 0
#     total_data = len(data)
#     for sent in data:
#         entities = sent[1]['entities']
#         doc = nlp(sent[0])
#         for token in doc:
#             #print(token.text, token.i, token.ent_iob_, token.ent_type_)
#             conll_entity = convert_spacy_entity_to_conll(
#                 token.ent_iob_, token.ent_type_)

#             # if match, update correct num
#             # note: entity[][0] contains token and entity[][1] contains iob-ent_type
#             if(conll_entity == entities[token.i][1]):
#                 result[conll_entity]['correct'] += 1

#             # update total num
#             result[conll_entity]['total'] += 1
#         index += 1
#         utils.printProgressBar(index, total_data,
#                                prefix='Progress calculating accuracy:', suffix='Complete', length=50)
#     return result

def get_refs_hyps(data):
    index = 0
    total_data = len(data)

    refs = []
    hyps = []

    for sent in data:
        entities = sent[1]['entities']
        hyp = []
        doc = nlp(sent[0])
        for token in doc:
            conll_entity = convert_spacy_entity_to_conll(
                token.ent_iob_, token.ent_type_)

            hyp.append((token.text, conll_entity))
        index += 1

        # add this sentence refs
        sent_refs = [(entity[0], entity[1]) for entity in entities]
        refs.append(sent_refs)
        # add this sent hyp
        hyps.append(hyp)

        utils.printProgressBar(index, total_data,
                               prefix='Progress calculating accuracy:', suffix='Complete', length=50)
    return refs, hyps


def main():
    # load data from file
    conll_test_data = load_conll_file(test_path)
    print('------------ First 2 sentences ------------')
    for k in conll_test_data[:2]:
        print('Sentence: ', k[0])
        print('Entities:')
        for e in k[1]['entities']:
            print(e[0], '|', e[1])

        print('************************************')

    print('\n')
    print('------------ Task 1.1: report token-level performance (per class and total) ------------')
    refs, hyps = get_refs_hyps(conll_test_data)
    accuracy = conll.evaluate(refs, hyps)
    print(accuracy)

    entity_refs = []
    for sent in refs:
        for token in sent:
            entity_refs.append(token[1])

    entity_hyps = []
    for sent in hyps:
        for token in sent:
            entity_hyps.append(token[1])

    confusion_matrix = ConfusionMatrix(entity_refs, entity_hyps)
    print(confusion_matrix)


if __name__ == '__main__':
    main()
