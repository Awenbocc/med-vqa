# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         model
# Description:  BAN model [Bilinear attention + Bilinear residual network]
# Author:       Boliu.Kelvin
# Date:         2020/4/7
#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from connect import FCNet
from connect import BCNet
from counting import Counter
from utils import tfidf_loading
from maml import SimpleCNN
from auto_encoder import Auto_Encoder_Model
from torch.nn.utils.weight_norm import weight_norm
from unet import Resnet50Encoder

# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):  # v:32,1,128; q:32,12,1024
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits

class BiResNet(nn.Module):
    def __init__(self,args,dataset,priotize_using_counter=False):
        super(BiResNet,self).__init__()
        # Optional module: counter
        use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10  # minimum number of boxes
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None
        # # init Bilinear residual network
        b_net = []   # bilinear connect :  (XTU)T A (YTV)
        q_prj = []   # output of bilinear connect + original question-> new question    Wq_ +q
        c_prj = []
        for i in range(args.glimpse):
            b_net.append(BCNet(dataset.v_dim, args.hid_dim, args.hid_dim, None, k=1))
            q_prj.append(FCNet([args.hid_dim, args.hid_dim], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, args.hid_dim], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.args = args

    def forward(self, v_emb, q_emb,att_p):
        b_emb = [0] * self.args.glimpse
        for g in range(self.args.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:]) # b x l x h
            # atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb.sum(1)

# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset,args):
        super(BAN_Model, self).__init__()

        self.args = args
        # init word embedding module, question embedding module, biAttention network, bi_residual network, and classifier
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.cat)
        self.q_emb = QuestionEmbedding(600 if args.cat else 300, args.hid_dim, 1, False, .0, args.rnn)
        self.bi_att = BiAttention(dataset.v_dim, args.hid_dim, args.hid_dim, args.glimpse)
        self.bi_resnet = BiResNet(args,dataset)
        self.classifier = SimpleClassifier(args.hid_dim, args.hid_dim * 2, dataset.num_ans_candidates, args)


        # build and load pre-trained MAML model
        if args.maml:
            weight_path = args.data_dir + '/' + args.maml_model_path
            print('load initial weights MAML from: %s' % (weight_path))
            self.maml = SimpleCNN(weight_path, args.eps_cnn, args.momentum_cnn)
        # build and load pre-trained Auto-encoder model
        if args.autoencoder:
            self.ae = Auto_Encoder_Model()
            weight_path = args.data_dir + '/' + args.ae_model_path
            print('load initial weights DAE from: %s' % (weight_path))
            self.ae.load_state_dict(torch.load(weight_path))
            self.convert = nn.Linear(16384, 64)
        # Loading tfidf weighted embedding
        if hasattr(args, 'tfidf'):
            self.w_emb = tfidf_loading(args.tfidf, self.w_emb, args)
            
        # Loading the other net
        if args.other_model:
            self.unet = Resnet50Encoder()
        

    def forward(self, v, q):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        # get visual feature
        if self.args.maml:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.args.other_model:
            v_emb = self.unet(v)  #input: b,c,h,w c==3 ; output= b,c,1,1
            v_emb = v_emb.squeeze(3).squeeze(2).unsqueeze(1) # b,1,c

        # get lextual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # Attention
        att_p, logits = self.bi_att(v_emb, q_emb) # b x g x v x q
        # bilinear residual network
        last_output = self.bi_resnet(v_emb,q_emb,att_p)
        if self.args.autoencoder:
                return last_output, decoder
        return last_output

    def classify(self, input_feats):
        return self.classifier(input_feats)


