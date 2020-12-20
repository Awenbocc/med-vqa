# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         process_dataset
# Description:  convert original .txt file to train.json and validate.json
# Author:       Boliu.Kelvin
# Date:         2020/4/5
#-------------------------------------------------------------------------------


import pandas as pd
import os
import sys
import json
import numpy as np
import re
import _pickle as cPickle

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}
manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer

def filter_answers(qa_pairs, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}

    for id, row in qa_pairs.iterrows(): # row:[id,ques,ans]
        gtruth = row['answer']
        gtruth = ' '.join(gtruth.split())
        # gtruth = preprocess_answer(gtruth)
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(row['question'])
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence

def create_ans2label(occurence,root='data'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    print('ans2lab', len(ans2label))
    print('lab2abs', len(label2ans))

    file = os.path.join(root, 'ans2label.pkl')
    cPickle.dump(ans2label, open(file, 'wb'))
    file = os.path.join(root, 'label2ans.pkl')
    cPickle.dump(label2ans, open(file, 'wb'))
    return ans2label

def compute_target(answers_dset, ans2label, name, root='data'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    count = 0
    for id,qa_pair in answers_dset.iterrows():
        answers = ' '.join(qa_pair['answer'].split())
        # answer_count = {}
        # for answer in answers:
        #     answer_ = answer['answer']
        #     answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        if answers in ans2label:
            scores.append(1.)
            labels.append(ans2label[answers])
        # for answer in answer_count:
        #     if answer not in ans2label:
        #         continue
        #     labels.append(ans2label[answer])
        #     score = get_score(answer_count[answer])
        #     scores.append(score)

        target.append({
            'question': qa_pair['question'],
            'image_name': qa_pair['id'],
            'labels': labels,
            'scores': scores
        })

    file = os.path.join(root, name+'_target.pkl')
    cPickle.dump(target, open(file, 'wb'))
    return target




if __name__ == '__main__' :
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    train_path = os.path.join(data,'VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_QA_pairs.txt')
    train_qa_pairs = pd.read_csv(train_path, sep='|', header=None, names=['id', 'question', 'answer'], index_col=None)
    occurence = filter_answers(train_qa_pairs, 0)  # select the answer with frequence over min_occurence

    label_path = data + 'ans2label.pkl'
    if os.path.isfile(label_path):
        print('found %s' % label_path)
        ans2label = cPickle.load(open(label_path, 'rb'))
    else:
        ans2label = create_ans2label(occurence,data)     # create ans2label and label2ans

    compute_target(train_qa_pairs, ans2label, 'train',data) #dump train target to .pkl {question,image_name,labels,scores}

    validate_path = os.path.join(data,'VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_QA_Pairs.txt')
    val_qa_pairs = pd.read_csv(validate_path, sep='|', header=None, names=['id', 'question', 'answer'], index_col=None)
    compute_target(val_qa_pairs, ans2label, 'validate', data)   #dump validate target to .pkl {question,image_name,labels,scores}
