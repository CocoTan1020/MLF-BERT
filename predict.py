# -- coding: utf-8 --
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, logging
from modeling_bert import BertModel
from char_word_gra_level import new_character, new_word, new_grammar, level_list2matrix
import numpy as np
from sklearn import metrics

logging.set_verbosity_warning()
logging.set_verbosity_error()
PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中综合信息符号



class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.save_path = './best_models_ckpt/mlf_bert/best_model.ckpt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', encoding='UTF-8').readlines()]
        self.device = torch.device('cpu')
        self.num_classes = len(self.class_list)
        self.pad_size = 512
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.only_max = True  # 语法等级是否取最大
        self.with_linguistic_information_embedding_layer = True
        self.with_linguistic_information_selfattention_layer = True

    def build_dataset(self, text):
        pad_size = self.pad_size
        token = config.tokenizer.tokenize(text)
        # 只在embedding层操作
        if self.with_linguistic_information_embedding_layer == True and self.with_linguistic_information_selfattention_layer == False:
            character_level_ids = new_character(token)
            word_level_ids = new_word(text, token)
            grammar_level_ids = new_grammar(text, token, only_max=config.only_max)
            character_level_ids.insert(0, 0)
            character_level_ids.append(0)
            word_level_ids.insert(0, 0)
            word_level_ids.append(0)
            grammar_level_ids.insert(0, 0)
            grammar_level_ids.append(0)
            if pad_size:
                temp = len(token) + 2
                if temp < pad_size:
                    character_level_ids += ([0] * (pad_size - temp))
                    word_level_ids += ([0] * (pad_size - temp))
                    grammar_level_ids += ([0] * (pad_size - temp))
                else:
                    character_level_ids = character_level_ids[:pad_size]
                    word_level_ids = word_level_ids[:pad_size]
                    grammar_level_ids = grammar_level_ids[:pad_size]

        # 只在selfattention层操作
        if self.with_linguistic_information_selfattention_layer == True and self.with_linguistic_information_embedding_layer == False:
            character_level_matrix = level_list2matrix(text, token, 'character')
            word_level_matrix = level_list2matrix(text, token, 'word')
            grammar_level_matrix = level_list2matrix(text, token, 'grammar')
            character_level_matrix = np.pad(character_level_matrix, ((1, 1), (1, 1)))
            word_level_matrix = np.pad(word_level_matrix, ((1, 1), (1, 1)))
            grammar_level_matrix = np.pad(grammar_level_matrix, ((1, 1), (1, 1)))
            if pad_size:
                temp = len(token) + 2
                if temp < pad_size:
                    character_level_matrix = np.pad(character_level_matrix,
                                                    ((0, pad_size - temp), (0, pad_size - temp)))
                    word_level_matrix = np.pad(word_level_matrix,
                                               ((0, pad_size - temp), (0, pad_size - temp)))
                    grammar_level_matrix = np.pad(grammar_level_matrix,
                                                  ((0, pad_size - temp), (0, pad_size - temp)))
                else:
                    character_level_matrix = character_level_matrix[:pad_size][:, :pad_size]
                    word_level_matrix = word_level_matrix[:pad_size][:, :pad_size]
                    grammar_level_matrix = grammar_level_matrix[:pad_size][:, :pad_size]

        # 既在embedding层操作，也在selfattention层操作
        if self.with_linguistic_information_embedding_layer == True and self.with_linguistic_information_selfattention_layer == True:
            character_level_ids = new_character(token)
            word_level_ids = new_word(text, token)
            grammar_level_ids = new_grammar(text, token, only_max=config.only_max)
            character_level_ids.insert(0, 0)
            character_level_ids.append(0)
            word_level_ids.insert(0, 0)
            word_level_ids.append(0)
            grammar_level_ids.insert(0, 0)
            grammar_level_ids.append(0)

            character_level_matrix = level_list2matrix(text, token, 'character')
            word_level_matrix = level_list2matrix(text, token, 'word')
            grammar_level_matrix = level_list2matrix(text, token, 'grammar')
            character_level_matrix = np.pad(character_level_matrix, ((1, 1), (1, 1)))
            word_level_matrix = np.pad(word_level_matrix, ((1, 1), (1, 1)))
            grammar_level_matrix = np.pad(grammar_level_matrix, ((1, 1), (1, 1)))

            if pad_size:
                temp = len(token) + 2
                if temp < pad_size:
                    character_level_ids += ([0] * (pad_size - temp))
                    word_level_ids += ([0] * (pad_size - temp))
                    grammar_level_ids += ([0] * (pad_size - temp))
                    character_level_matrix = np.pad(character_level_matrix,
                                                    ((0, pad_size - temp), (0, pad_size - temp)))
                    word_level_matrix = np.pad(word_level_matrix,
                                               ((0, pad_size - temp), (0, pad_size - temp)))
                    grammar_level_matrix = np.pad(grammar_level_matrix,
                                                  ((0, pad_size - temp), (0, pad_size - temp)))
                else:
                    character_level_ids = character_level_ids[:pad_size]
                    word_level_ids = word_level_ids[:pad_size]
                    grammar_level_ids = grammar_level_ids[:pad_size]
                    character_level_matrix = character_level_matrix[:pad_size][:, :pad_size]
                    word_level_matrix = word_level_matrix[:pad_size][:, :pad_size]
                    grammar_level_matrix = grammar_level_matrix[:pad_size][:, :pad_size]

        token = [CLS] + token + [SEP]
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size

        # 在bert-embedding层加入等级信息
        if self.with_linguistic_information_embedding_layer and self.with_linguistic_information_selfattention_layer == False:
            return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask]), torch.tensor([character_level_ids], dtype=torch.long), torch.tensor([word_level_ids], dtype=torch.long), torch.tensor([grammar_level_ids], dtype=torch.long)
        # 在bert-self-attention层加入等级信息
        if self.with_linguistic_information_selfattention_layer and self.with_linguistic_information_embedding_layer == False:
            return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask]), torch.tensor([character_level_matrix], dtype=torch.long), torch.tensor([word_level_matrix], dtype=torch.long), torch.tensor([grammar_level_matrix], dtype=torch.long)
        # 普通的bert
        if self.with_linguistic_information_embedding_layer == False and self.with_linguistic_information_selfattention_layer == False:
            return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask])

        # 既在embedding层操作，也在selfattention层操作
        if self.with_linguistic_information_embedding_layer and self.with_linguistic_information_selfattention_layer:
            return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask]), torch.tensor(
                [character_level_ids], dtype=torch.long), torch.tensor([word_level_ids],
                                                                       dtype=torch.long), torch.tensor(
                [grammar_level_ids], dtype=torch.long), torch.tensor([character_level_matrix], dtype=torch.long), torch.tensor([word_level_matrix], dtype=torch.long), torch.tensor([grammar_level_matrix], dtype=torch.long)


class Model(nn.Module):
    """配置模型"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, output_attentions=True)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        token_type_ids = torch.zeros(x[0].size()[0], x[0].size()[1]).to(x[0].device)
        if config.with_linguistic_information_embedding_layer == False and config.with_linguistic_information_selfattention_layer == False:
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long())

        # 在bert的embedding层加入等级特征
        if config.with_linguistic_information_embedding_layer and config.with_linguistic_information_selfattention_layer == False:
            character_level = x[2]
            word_level = x[3]
            grammar_level = x[4]
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long(),
                             character_level_ids=character_level, word_level_ids=word_level,
                             grammar_level_ids=grammar_level)

        # 在bert的selfattention层加入等级特征
        if config.with_linguistic_information_selfattention_layer and config.with_linguistic_information_embedding_layer == False:
            character_level = x[2]
            word_level = x[3]
            grammar_level = x[4]
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long(),
                             character_level_matrix=character_level, word_level_matrix=word_level,
                             grammar_level_matrix=grammar_level)

        # 既在bert的embedding层加入等级特征，又在bert的selfattention层加入等级特征
        if config.with_linguistic_information_embedding_layer and config.with_linguistic_information_selfattention_layer:
            character_level_ids = x[2]
            word_level_ids = x[3]
            grammar_level_ids = x[4]
            character_level_matrix = x[5]
            word_level_matrix = x[6]
            grammar_level_matrix = x[7]
            outs = self.bert(input_ids=context, attention_mask=mask, token_type_ids=token_type_ids.long(),
                             character_level_ids=character_level_ids, word_level_ids=word_level_ids,
                             grammar_level_ids=grammar_level_ids, character_level_matrix=character_level_matrix,
                             word_level_matrix=word_level_matrix, grammar_level_matrix=grammar_level_matrix)

        pooled = outs[1]
        out = self.fc(pooled)
        return outs, out

def load_model():
    dataset = 'hsk_all'
    config = Config(dataset=dataset)
    model = Model(config).to(config.device)
    print('Loading classification model...')
    model.load_state_dict(torch.load(config.save_path, map_location='cpu'))
    # print(model)
    # for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        # print(parameters)
    return config, model

def prediction_model(config, model, text):
    """输入text预测"""
    data = config.build_dataset(text)
    class_list = config.class_list
    with torch.no_grad():
        outs, outputs = model(data)
        # num = torch.argmax(outputs)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
    return int(predic), class_list[int(predic)], outs

if __name__ == '__main__':

    text = '''在电子商务风行的今天，实体店还能多大程度地影响品牌在零售市场的地位?
    就在零售商都为实体店的未来捏把汗的时候，有人做了一项调查，令人惊讶的是，未来计划更多通过实体店购物的消费者比例从一年前的18%攀升至26%；表示实体店"方便购物"的客户达到93%，远高于网络和移动设备。
    这似乎与许多零售商的认识相去甚远。
    过去一年多里，传统零售商纷纷扎堆规划电子商务，由店商向电商转型。
    根据调查，63%的传统零售商已开展多渠道零售，但近三成零售商表示，其多渠道战略实施并不成功。
    究其根本，是因为许多零售企业并未深入了解消费者的需求变化，其转型初衷只是为了数字化而数字化，认为仅仅通过技术的部署就能带来绩效的提升。
    事实上，在数字时代，虽然技术的变革重新定义了零售商与消费者的连接方式，但却并没有改变消费者需求的本质是价格合理、产品种类丰富以及多年积累的信任感。
    因此，掌控零售商未来命运的，不是涌现的新兴技术，更不是"凶猛"的互联网电商，而是瞬息万变的消费者需求。'''


    # 载入模型
    config, model = load_model()
    # 输出预测等级
    _, pred_level, _ = prediction_model(config, model, text)
    print(pred_level)


