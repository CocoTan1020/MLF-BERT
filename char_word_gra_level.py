# coding: UTF-8
import re
import jieba
import numpy as np
import json
jieba.setLogLevel(jieba.logging.INFO)
np.set_printoptions(threshold=np.inf)


def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w', encoding='utf-8')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r', encoding='gbk')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def json2list(path):
    with open(path, "r", encoding='utf-8') as f:
        dict = json.load(f)
        word_len = list(dict['word'])
        level_len = list(dict['level'])
    return word_len, level_len

def load_dict_json(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


# 读入new汉字分级大纲
new_level1_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level1.txt')
new_level2_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level2.txt')
new_level3_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level3.txt')
new_level4_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level4.txt')
new_level5_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level5.txt')
new_level6_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level6.txt')
new_level7_character_list = list_txt(path ='NEW_outline/new_character_level1-7/level7.txt')

# 读入new词汇分级大纲
new_level1_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level1.txt')
new_level2_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level2.txt')
new_level3_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level3.txt')
new_level4_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level4.txt')
new_level5_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level5.txt')
new_level6_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level6.txt')
new_level7_word_list = list_txt(path ='NEW_outline/new_word_level1-7/level7.txt')

# 读入new词汇大纲分长度的json格式
new_word_len1, new_level_len1 = json2list(path="NEW_outline/new_word_level1-7/word_len1_level.json")
new_word_len2, new_level_len2 = json2list(path="NEW_outline/new_word_level1-7/word_len2_level.json")
new_word_len3, new_level_len3 = json2list(path="NEW_outline/new_word_level1-7/word_len3_level.json")
new_word_len4, new_level_len4 = json2list(path="NEW_outline/new_word_level1-7/word_len4_level.json")

# 读入语法点正则函数
word_patterns_1 = load_dict_json('NEW_outline/new_grammar_level1-7/gram_word_patterns_1.0.json')
sent_patterns_1 = load_dict_json('NEW_outline/new_grammar_level1-7/gram_sent_patterns_1.0.json')

word_patterns_2 = load_dict_json('NEW_outline/new_grammar_level1-7/gram_word_patterns_2.0.json')
sent_patterns_2 = load_dict_json('NEW_outline/new_grammar_level1-7/gram_sent_patterns_2.0.json')

# 读入语法点非连续匹配公式
gram_discrete = load_dict_json('NEW_outline/new_grammar_level1-7/gram_discrete_2.0.json')


def judge_grammar_level(i):
    if (1 <= i <= 48):
        return 1
    if (49 <= i <= 130):
        return 2
    if (131 <= i <= 211):
        return 3
    if (212 <= i <= 287):
        return 4
    if (288 <= i <= 358):
        return 5
    if (359 <= i <= 425):
        return 6
    if (426 <= i <= 573):
        return 7

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def token_c_begin(token, begin, length):
    temp = []
    count = 0
    for i in range(begin, len(token)):
        if '\u4e00' <= token[i] <= '\u9fff':
            count += 1
            temp.append(token[i])
        if count == length:
            temp_sentence = ''.join(temp)
            return temp_sentence

def token_c_back(token, back, length):
    token = [token[len(token) - i - 1] for i in range(len(token))]
    temp_sentence = token_c_begin(token, back, length)
    return temp_sentence

def find_sentence_index(sentence, token):
    sentence_ = find_chinese(sentence)
    if len(sentence_) >= 5:
        find_prefix = sentence_[:5]
        find_suffix = sentence_[-5:]
    else:
        find_prefix = sentence_
        find_suffix = sentence_
    index1 = 0
    find_index1 = False
    for i in range(len(token)):
        temp = token_c_begin(token, i, len(find_prefix))

        if temp == find_prefix:
            index1 = i
            find_index1 = True
            break
    index2 = index1
    find_index2 = False
    for i in range(len(token)):
        temp = token_c_back(token, i, len(find_suffix))
        # print(temp)
        if temp == None:
            break
        temp = temp[::-1]
        if temp == find_suffix:
            find_index2 = True
            index2 = len(token) - i
            break
    if find_index1 == False or find_index2 == False:
        index1, index2 = 0, 0
    tt = token[index1:index2].copy()

    # 解决以非中文字开头和结尾的特殊情况
    if token[index1] > '\u9fff' or token[index1] < '\u4e00' and len(tt) > 1:
        index1 += 1
    if token[index2 - 1] > '\u9fff' or token[index2 - 1] < '\u4e00' and len(tt) > 1:
        index2 -= 1
    return index1, index2

def find_word_index(word, token):
    index1_list = []
    index2_list = []
    word_length = len(word)
    for i in range(len(token)):
        temp = ''.join(token[i:i+word_length])
        if temp == word:
            index1_list.append(i)
            index2_list.append(i+word_length)
    return index1_list, index2_list

def cut(text):
    pattern = ['([。！？\?])([^”’])','(\.{6})([^”’])','(\…{2})([^”’])','([。！？\?][”’])([^，。！？\?])']
    for i in pattern:
        text = re.sub(i, r"\1\n\2", text)
    text = text.rstrip()
    return text.split("\n")

def text2sentence(text):
    sentence_before = cut(text)
    # 将被错分的语句进行连接
    list = []
    sentence = ""
    FLAG = True  # 判断有'：“'的符号后面的语句是否继续拼接
    for i in sentence_before:
        if i == '':
            continue
        if i.find('：“') * i.find('”') >= 0 and FLAG:
            list.append(i + sentence)
        else:
            FLAG = False
            sentence = sentence + i
            if i.find('”') > 0:
                list.append(sentence)
                sentence = ""
                FLAG = True
    sentence_after = list
    return sentence_after

def word_different_len_select(text):
    word_list = []
    word_level_list = []
    for word_len in (1,2,3,4):
        for i in range(len(text)-word_len+1):
            temp = text[i:i+word_len]
            if temp in eval('new_word_len' + str(word_len)):
                word_list.append(temp)
                temp_index = eval('new_word_len' + str(word_len)).index(temp)
                temp_level = eval('new_level_len' + str(word_len))[temp_index]
                word_level_list.append(temp_level)
    return word_list, word_level_list

def new_word_ngram(text, token, ngram_way='max_len', for_matrix=False):
    '''
    way：max_level or max_len，defult=max_len
    '''
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
    cut_word_list, new_word_list_ = word_different_len_select(text)
    new_word_list = [0 for i in token]
    start_index = [0 for i in token]
    end_index = [0 for i in token]
    for w in range(len(cut_word_list)):
        index1_list, index2_list = find_word_index(cut_word_list[w], token)
        for i, j in zip(index1_list, index2_list):
            for k in range(i, j):
                # 保留该token的最高等级标签
                if ngram_way == 'max_level':
                    if new_word_list_[w] > new_word_list[k]:
                        new_word_list[k] = new_word_list_[w]
                        start_index[k] = i
                        end_index[k] = j
                # 保留最长匹配长度的等级标签
                if ngram_way == 'max_len':
                    new_word_list[k] = new_word_list_[w]
                    start_index[k] = i
                    end_index[k] = j
    assert len(new_word_list) == len(token)
    if for_matrix == False:
        return new_word_list
    if for_matrix == True:
        assert len(start_index) == len(new_word_list)
        return (start_index, end_index, new_word_list)

def gram_discrete_find(gram_no, text):
    if str(gram_no) in gram_discrete.keys():
        for i in eval(gram_discrete[str(gram_no)]):
            flag = True
            for g in i:
                if g not in text:
                    flag = False
            if flag:
                return i
    return []

def new_grammar_with_discrete(text, token, only_max=True, version=2.0, for_matrix=False):
    if version == 1.0:
        word_patterns = word_patterns_1
        sent_patterns = sent_patterns_1
    if version == 2.0:
        word_patterns = word_patterns_2
        sent_patterns = sent_patterns_2
    text = text.replace('\n', '').replace('\r', '')
    seg_list = jieba.cut(text, cut_all=False)
    cut_word_list = []
    for word in seg_list:
        cut_word_list.append(word)

    grammar = []
    grammar_level = []
    match_begin = []
    match_end = []
    # 找出单词层级的语法
    for j in range(len(cut_word_list)):
        for i in word_patterns:
            if (re.search(word_patterns[i], cut_word_list[j])):
                match = re.search(word_patterns[i], cut_word_list[j]).group()
                level = judge_grammar_level(int(i))
                grammar.append(match)
                grammar_level.append(level)
                match_begin.append(0)
                match_end.append(len(token))
    # 找出句子层级的语法
    tt = '\n'.join(text2sentence(text))
    for i in sent_patterns:
        if (re.search(sent_patterns[i], tt)):
            match = re.search(sent_patterns[i], tt).group()
            level = judge_grammar_level(int(i))
            gram_discrete_list = gram_discrete_find(i, match)
            if gram_discrete_list == []:
                grammar.append(match)
            else:
                grammar.append(gram_discrete_list)
            grammar_level.append(level)
            index1, index2 = find_sentence_index(match, token)
            match_begin.append(index1)
            match_end.append(index2)

    temp_grammar = []
    temp_level = []
    for i in range(len(grammar)):
        if type(grammar[i]) != type([]):
            ttt = []
            index1_list, index2_list = find_word_index(grammar[i], token)
            for p in range(len(index1_list)):
                temp = [index1_list[p], index2_list[p]]
                ttt.append(temp)
            temp_grammar.append(ttt)
            temp_level.append(grammar_level[i])
        else:
            ttt = []
            for gd in grammar[i]:
                index1_list, index2_list = find_word_index(gd, token[match_begin[i]:match_end[i]+1])
                if index1_list == [] or index2_list == []:
                    continue
                temp = [match_begin[i] + index1_list[0], match_begin[i] + index2_list[0]]
                ttt.append(temp)
            temp_grammar.append(ttt)
            temp_level.append(grammar_level[i])

    # 每一个token位置都是一个列表
    if only_max == False:
        return_list_list = []
        for i in range(len(token)):
            return_list_list.append([])
        for i in range(len(temp_level)):
            level = temp_level[i]
            index_list = temp_grammar[i]
            for ii in index_list:
                begin = ii[0]
                end = ii[1]
                for p in range(ii[0], ii[1]):
                    return_list_list[p].append(level)

        assert len(return_list_list) == len(token)
        return_list = return_list_list
    # 每一个token位置都是最高的语法等级
    if only_max == True:
        return_list_list = []
        for i in range(len(token)):
            return_list_list.append(0)
        for i in range(len(temp_level)):
            level = temp_level[i]
            index_list = temp_grammar[i]
            for ii in index_list:
                begin = ii[0]
                end = ii[1]
                for p in range(ii[0], ii[1]):
                    if return_list_list[p] < level:
                        return_list_list[p] = level
        assert len(return_list_list) == len(token)
        return_list = return_list_list
    if for_matrix == False:
        return return_list
    # 返回语法点邻接矩阵所需数据
    if for_matrix == True:
        return (temp_grammar, temp_level)

def level_list2matrix(text, token, level, way='level'):
    level_m = np.zeros((len(token), len(token)), dtype=np.int64)
    if level == 'character' or level == 'word':
        if level == 'character':
            index_list = new_character(token, for_matrix=True)
        elif level == 'word':
            index_list = new_word(text, token, for_matrix=True)
            # print(index_list)
        for i in range(len(index_list[2])):
            s = index_list[0][i]
            e = index_list[1][i]
            # print(text[s:e])
            for ii in range(s, e):
                for jj in range(s, e):
                    # 方式1：0，1矩阵
                    if way == '01':
                        level_m[ii, jj] = 1
                    # 方式2：0，level矩阵
                    if way == 'level':
                        if index_list[2][i] > level_m[ii, jj]:
                            level_m[ii, jj] = index_list[2][i]
        return level_m
    # 对语法矩阵进行特殊处理
    if level == 'grammar':
        index_list = new_grammar(text, token, for_matrix=True)
        for i in range(len(index_list[1])):
            tt = index_list[0][i]
            if len(tt) == 1:
                s1 = tt[0][0]
                e1 = tt[0][1]
                for ii in range(s1, e1):
                    for jj in range(s1, e1):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
            elif len(tt) == 2:
                s1 = tt[0][0]
                e1 = tt[0][1]
                s2 = tt[1][0]
                e2 = tt[1][1]
                for ii in range(s1, e1):
                    for jj in range(s1, e1):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s2, e2):
                    for jj in range(s2, e2):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s1, e1):
                    for jj in range(s2, e2):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s2, e2):
                    for jj in range(s1, e1):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
            if len(tt) == 3:
                s1 = tt[0][0]
                e1 = tt[0][1]
                s2 = tt[1][0]
                e2 = tt[1][1]
                s3 = tt[2][0]
                e3 = tt[2][1]
                for ii in range(s1, e1):
                    for jj in range(s1, e1):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s2, e2):
                    for jj in range(s2, e2):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s3, e3):
                    for jj in range(s3, e3):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s1, e1):
                    for jj in range(s2, e2):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s2, e2):
                    for jj in range(s1, e1):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s1, e1):
                    for jj in range(s3, e3):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s2, e2):
                    for jj in range(s3, e3):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s3, e3):
                    for jj in range(s1, e1):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
                for ii in range(s3, e3):
                    for jj in range(s2, e2):
                        if way == '01':
                            level_m[ii, jj] = 1
                        # 方式2：0，level矩阵
                        if way == 'level':
                            if index_list[1][i] > level_m[ii, jj]:
                                level_m[ii, jj] = index_list[1][i]
        return level_m

def new_character(token, for_matrix=False):
    new_character_list = []
    for i in range(len(token)):
        new_character_list.append(0)
        for j in range(7):
            if token[i] in eval('new_level' + str(j+1) + '_character_list'):
                new_character_list[i] += j+1
                break
    assert len(new_character_list) == len(token)
    if for_matrix == False:
        return new_character_list
    else:
        start_index = [i for i in range(len(token))]
        end_index = [i + 1 for i in range(len(token))]
        return (start_index, end_index, new_character_list)

def new_word(text, token, way='ngram_max_len', for_matrix=False):
    if way == 'ngram_max_level':
        if for_matrix == False:
            new_word_list = new_word_ngram(text, token, ngram_way='max_level', for_matrix=False)
            return new_word_list
        else:
            level_list = new_word_ngram(text, token, ngram_way='max_level', for_matrix=True)
            return level_list
    elif way == 'ngram_max_len':
        if for_matrix == False:
            new_word_list = new_word_ngram(text, token, ngram_way='max_len', for_matrix=False)
            return new_word_list
        else:
            level_list = new_word_ngram(text, token, ngram_way='max_len', for_matrix=True)
            return level_list
    else:
        print('不支持')

def new_grammar(text, token, only_max=True, version=2.0, for_matrix=False, discrete=True):
    if discrete and not for_matrix:
        return_list = new_grammar_with_discrete(text, token, only_max=only_max, version=version, for_matrix=for_matrix)
        return return_list
    elif discrete and for_matrix:
        return_list = new_grammar_with_discrete(text, token, only_max=only_max, version=version, for_matrix=for_matrix)
        return return_list
    else:
        print('不支持')
