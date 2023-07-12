# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer
from modeling_bert import BertModel

class Config(object):

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='UTF-8').readlines()]
        self.save_path = 'saved_dict/'
        self.save_epoch = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_list)
        self.num_epochs = 20
        self.batch_size = 32
        self.pad_size = 512
        self.learning_rate = 5e-5
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.only_max = True  # 语法等级是否取最大
        self.with_linguistic_information_embedding_layer = True  # 是否在bertembedding层融入语言特征
        self.with_linguistic_information_selfattention_layer = True  # 是否在bertselfattention层融入语言特征


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.with_linguistic_information_embedding_layer = config.with_linguistic_information_embedding_layer
        self.with_linguistic_information_selfattention_layer = config.with_linguistic_information_selfattention_layer

    def forward(self, x):
        context = x[0]
        mask = x[2]
        token_type_ids = torch.zeros(x[0].size()[0], x[0].size()[1]).to(x[0].device)

        if not self.with_linguistic_information_embedding_layer and \
                not self.with_linguistic_information_selfattention_layer:
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long())

        if self.with_linguistic_information_embedding_layer and \
                not self.with_linguistic_information_selfattention_layer:
            character_level = x[3]
            word_level = x[4]
            grammar_level = x[5]
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long(),
                             character_level_ids=character_level, word_level_ids=word_level,
                             grammar_level_ids=grammar_level)

        if self.with_linguistic_information_selfattention_layer and \
                not self.with_linguistic_information_embedding_layer:
            character_level = x[3]
            word_level = x[4]
            grammar_level = x[5]
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long(),
                             character_level_matrix=character_level, word_level_matrix=word_level,
                             grammar_level_matrix=grammar_level)

        if self.with_linguistic_information_embedding_layer and \
                self.with_linguistic_information_selfattention_layer:
            character_level_ids = x[3]
            word_level_ids = x[4]
            grammar_level_ids = x[5]
            character_level_matrix = x[6]
            word_level_matrix = x[7]
            grammar_level_matrix = x[8]
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long(),
                             character_level_ids=character_level_ids, word_level_ids=word_level_ids,
                             grammar_level_ids=grammar_level_ids, character_level_matrix=character_level_matrix,
                             word_level_matrix=word_level_matrix, grammar_level_matrix=grammar_level_matrix)
        pooled = outs[1]
        out = self.fc(pooled)
        return out
