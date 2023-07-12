# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from char_word_gra_level import new_character, new_word, new_grammar, level_list2matrix
import numpy as np


PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)

                # 只在embedding层操作
                if config.with_linguistic_information_embedding_layer == True and config.with_linguistic_information_selfattention_layer == False:
                    character_level_ids = new_character(token)
                    word_level_ids = new_word(content, token)
                    grammar_level_ids = new_grammar(content, token, only_max=config.only_max)
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
                if config.with_linguistic_information_selfattention_layer == True and config.with_linguistic_information_embedding_layer == False:
                    character_level_matrix = level_list2matrix(content, token, 'character')
                    word_level_matrix = level_list2matrix(content, token, 'word')
                    grammar_level_matrix = level_list2matrix(content, token, 'grammar')
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
                if config.with_linguistic_information_embedding_layer == True and config.with_linguistic_information_selfattention_layer == True:
                    character_level_ids = new_character(token)
                    word_level_ids = new_word(content, token)
                    grammar_level_ids = new_grammar(content, token, only_max=config.only_max)
                    character_level_ids.insert(0, 0)
                    character_level_ids.append(0)
                    word_level_ids.insert(0, 0)
                    word_level_ids.append(0)
                    grammar_level_ids.insert(0, 0)
                    grammar_level_ids.append(0)

                    character_level_matrix = level_list2matrix(content, token, 'character')
                    word_level_matrix = level_list2matrix(content, token, 'word')
                    grammar_level_matrix = level_list2matrix(content, token, 'grammar')
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
                if config.with_linguistic_information_embedding_layer and config.with_linguistic_information_selfattention_layer == False:
                    contents.append(
                        (token_ids, int(label), seq_len, mask, character_level_ids, word_level_ids, grammar_level_ids))
                # 在bert-self-attention层加入等级信息
                if config.with_linguistic_information_selfattention_layer and config.with_linguistic_information_embedding_layer == False:
                    contents.append((token_ids, int(label), seq_len, mask, character_level_matrix, word_level_matrix,
                                     grammar_level_matrix))
                # 普通的bert
                if config.with_linguistic_information_embedding_layer == False and config.with_linguistic_information_selfattention_layer == False:
                    contents.append((token_ids, int(label), seq_len, mask))

                # 既在embedding层操作，也在selfattention层操作
                if config.with_linguistic_information_embedding_layer and config.with_linguistic_information_selfattention_layer:
                    contents.append((token_ids, int(label), seq_len, mask, character_level_ids, word_level_ids, grammar_level_ids,
                                     character_level_matrix, word_level_matrix, grammar_level_matrix))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # for普通的bert
        if len(datas[0]) == 4:
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        # for 只在bert-embedding层加入等级信息或者只在bert-self-attention层加入等级信息
        elif len(datas[0]) == 7:
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            character_level = torch.LongTensor([_[4] for _ in datas]).to(self.device)
            word_level = torch.LongTensor([_[5] for _ in datas]).to(self.device)
            grammar_level = torch.LongTensor([_[6] for _ in datas]).to(self.device)
            return (x, seq_len, mask, character_level, word_level, grammar_level), y

        # for 既在bert-embedding层加入等级信息，又在bert-self-attention层加入等级信息
        elif len(datas[0]) == 10:
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            character_level_ids = torch.LongTensor([_[4] for _ in datas]).to(self.device)
            word_level_ids = torch.LongTensor([_[5] for _ in datas]).to(self.device)
            grammar_level_ids = torch.LongTensor([_[6] for _ in datas]).to(self.device)
            character_level_matrix = torch.LongTensor([_[7] for _ in datas]).to(self.device)
            word_level_matrix = torch.LongTensor([_[8] for _ in datas]).to(self.device)
            grammar_level_matrix = torch.LongTensor([_[9] for _ in datas]).to(self.device)
            return (x, seq_len, mask, character_level_ids, word_level_ids, grammar_level_ids, character_level_matrix, word_level_matrix, grammar_level_matrix), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
