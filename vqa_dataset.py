# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         vqa_dataset
# Description:  VQAdataset : vision{maml + autoencoder} & questions & labels
# Author:       Boliu.Kelvin
# Date:         2020/4/6
#-------------------------------------------------------------------------------

import _pickle as cPickle
import numpy as np
from torch.utils.data import Dataset
import os
import utils
from PIL import Image
import torch
import torchvision.transforms as transforms

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data', question_len=12):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        assert name in ['train', 'validate']
        ans2label_path = os.path.join(dataroot,  'ans2label.pkl')
        print(ans2label_path)
        label2ans_path = os.path.join(dataroot,  'label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        # Get the word dictionary
        self.dictionary = dictionary
        #Get the target [question,image_name,labels,scores] of [trian or validate]
        self.entries = cPickle.load(open(os.path.join(dataroot,name+'_target.pkl'), 'rb'))

        if name =='train':
            self.images_path = os.path.join(dataroot, 'VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_images')
        elif name =='validate':
            self.images_path = os.path.join(dataroot, 'VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_images')

        self.tokenize(question_len)
        # self.tensorize()
        if args.autoencoder and args.maml:
            self.v_dim = args.v_dim * 2
        if args.other_model:
            self.v_dim = args.v_dim
            #self.v_dim = 3904
            

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens


    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        image_name = entry['image_name']
        
        if self.args.maml and self.args.autoencoder:
            image_data = [0,0]
            
        if self.args.other_model:
            image_data = None
            

        if self.args.maml:
            maml_compose = transforms.Compose([
                transforms.Resize([84, 84]),
                transforms.ToTensor()
            ])
            maml_images_data = Image.open(os.path.join(self.images_path,image_name)+'.jpg').convert('L') #gray level pic
            image_data[0] = maml_compose(maml_images_data)


        if self.args.autoencoder:

            ae_compose = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])
            ae_images_data = Image.open(os.path.join(self.images_path, image_name) + '.jpg').convert('L')
            image_data[1] = ae_compose(ae_images_data)
        
        if self.args.other_model:
            compose = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
            images_data = Image.open(os.path.join(self.images_path, image_name) + '.jpg')
            image_data = compose(images_data)
            

        labels = np.array(entry['labels'])
        scores = np.array(entry['scores'],dtype=np.float32)
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, torch.tensor(labels), torch.tensor(scores))

        return  image_data,np.array(question),target



    def __len__(self):
        return len(self.entries)


