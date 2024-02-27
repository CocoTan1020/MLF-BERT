# MLF-BERT 
# 基于多层级语言特征融合的中文文本可读性分级模型

这是该论文的代码仓库: **《基于多层级语言特征融合的中文文本可读性分级模型》** (中文信息学报2024-待刊出)。

**论文的Preprint版本可以在该链接中进行查看。** [[基于多层级语言特征融合的中文文本可读性分级模型-Preprint]](https://drive.google.com/file/d/1i7iNjABG0I_w0gfKk-0v6uDkGGd_mKxX/view?usp=sharing)

如果您发现这项工作对您的研究有用，请引用我们的论文：
> @article{谭可人2023基于多层级语言特征融合的中文文本可读性分级模型,  
>   title={基于多层级语言特征融合的中文文本可读性分级模型},  
>   author={谭可人 and 兰韵诗 and 张杨 and 丁安琪},  
>   journal={中文信息学报},  
>   volume={-},  
>   number={-},  
>   pages={-},  
>   year={2024}  
> }

## Installation
运行以下代码以安装所需的库:
```
pip install -r requirements.txt
```
使用Python 3.7环境
## Datasets
数据集来自于中文文本可读性分级数据集[[CTRDG]](https://github.com/CocoTan1020/CTRDG/tree/main)。
```
训练集：hsk_all/data/train.txt

验证集：hsk_all/data/dev.txt

测试集：hsk_all/data/test.txt
```
## Pretrained Models
中文BERT预训练模型下载地址：[[BERT]](https://huggingface.co/bert-base-chinese)
```
中文BERT预训练模型配置文件：bert_pretrain/config.json

中文BERT预训练模型参数文件：bert_pretrain/pytorch_model.bin

中文BERT预训练模型词表文件：bert_pretrain/vocab.txt
```

## Train model
#### 1. 在```bert_pretrain/config.json```文件中添加以下配置信息
```
{
  (configurations about BERT Pretrained Model)
  ......

  "with_linguistic_information_embedding_layer": "True",
  "with_character_level_embedding_layer": "True",
  "with_word_level_embedding_layer": "True",
  "with_grammar_level_embedding_layer": "False",
  "character_level_size_embedding_layer": 8,
  "word_level_size_embedding_layer": 8,
  "grammar_level_size_embedding_layer": 8,

  "with_linguistic_information_selfattention_layer" : "True",
  "linguistic_information_selfattention_layer_num": 7,
  "with_character_level_selfattention_layer": "False",
  "with_word_level_selfattention_layer": "False",
  "with_grammar_level_selfattention_layer": "True",
  "character_level_hp_selfattention_layer": 1,
  "word_level_hp_selfattention_layer": 1,
  "grammar_level_hp_selfattention_layer": 1,
  "level_with_nnembedding": "False",
  "add_begin_attention_layer": 0
}
```
**Note:**

- ```with_linguistic_information_embedding_layer```  是否在Bertembedding层添加语言特征
- ```with_character/word/grammar_level_embedding_layer```  是否在Bertembedding层添加汉字/词汇/语法语言特征
- ```character_level_size_embedding_layer```  汉字/词汇/语法语言特征的等级数量 (大纲等级数量为7，添加1维作为未匹配等级)
- ```with_linguistic_information_selfattention_layer```  是否在Bertselfattention层添加语言特征
- ```linguistic_information_selfattention_layer_num```  添加语言特征的Bertselfattention层数量
- ```with_character/word/grammar_level_selfattention_layer```  是否在Bertselfattention层添加汉字/词汇/语法语言特征
- ```character/word/grammar_level_hp_selfattention_layer```  在Bertselfattention层中添加汉字/词汇/语法语言特征的权重超参数
- ```level_with_nnembedding```  在Bertselfattention层中是否对特征的等级进行embedding处理
- ```add_begin_attention_layer```  设置添加语言特征的起始Bertselfattention层编号
#### 2. 在```bert.py```文件中进行模型配置
注意保持以下两个字段与```bert_pretrain/config.json```文件的一致性。
```
self.with_linguistic_information_embedding_layer = True  # 是否在bertembedding层融入语言特征
self.with_linguistic_information_selfattention_layer = True  # 是否在bertselfattention层融入语言特征
```
#### 3. 开始模型训练
使用以下代码对模型进行训练。
```
python run.py
```
## Model Inference
使用以下代码对输入的文本进行可读性等级预测。
```
python predict.py
```
## LLM Evaluation Method
```
python llm_evaluation/MLF-BERT_llm_evaluate.py
```
## Well-Trained Model
训练好的MLF-BERT模型可以在该链接中下载获取  [[MLF-BERT]](https://pan.baidu.com/s/1uHKWq_7FKEJJsmmocg3bcQ?pwd=h4sp)
| Model     | Test_Acc/%  | Test_F1/%   |
| -------- | -------- | -------- |
| BERT |91.10 | 90.97 |
| MLF-BERT | **94.24** | **93.96** |


