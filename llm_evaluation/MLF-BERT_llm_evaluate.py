import os
from openai import OpenAI
import json
import re
from http import HTTPStatus
import dashscope
from dashscope import Generation
from sklearn import metrics

dashscope.api_key = ""
os.environ["OPENAI_API_KEY"] = ""

prompt = '''
根据以下HSK标准评估文本的难易程度：

1级：能理解并使用一些非常简单的汉语词语和句子，具备进一步学习汉语能力。\\
2级：能用汉语就生活中一些常见的话题进行简单而直接的交流。\\
3级：能用汉语完成生活、学习、工作等方面的基本交际任务。\\
4级：能用汉语就比较复杂的话题进行交流，表达较为规范、得体。\\
5级：能用汉语就比较抽象或专业的话题进行讨论、评价和发表看法，能较轻松地应对各种交际任务。\\
6级：能用汉语自如地进行各种社会交际活动，汉语应用水平接近汉语为母语者。

文本如下：
[text]
下面写出该文本的难度等级(1-6级)：
'''

prompt_6shot = '''
根据以下HSK标准评估文本的难易程度：

1级：能理解并使用一些非常简单的汉语词语和句子，具备进一步学习汉语能力。\\
2级：能用汉语就生活中一些常见的话题进行简单而直接的交流。\\
3级：能用汉语完成生活、学习、工作等方面的基本交际任务。\\
4级：能用汉语就比较复杂的话题进行交流，表达较为规范、得体。\\
5级：能用汉语就比较抽象或专业的话题进行讨论、评价和发表看法，能较轻松地应对各种交际任务。\\
6级：能用汉语自如地进行各种社会交际活动，汉语应用水平接近汉语为母语者。

例如：
文本：你是哪国人？ 中国人。
下面写出该文本的难度等级(1-6级)：1级
文本：我觉得今天天气不太好，很可能会下雨。 那打电话告诉他别去踢球了，明天再去。
下面写出该文本的难度等级(1-6级)：2级
文本：中国人常说：六月的天，孩子的脸，一日变三变。意思是夏天天气变化快可能上午还是晴天，下午就阴天下雨了。
下面写出该文本的难度等级(1-6级)：3级
文本：“生活中不缺少美，缺少的是发现美的眼睛。”草绿了，那是生命的颜色；花开了，那是大自然的礼物。只要有心，生活中的美到处都是。生活的态度由自己选择。
下面写出该文本的难度等级(1-6级)：4级
文本：在衣食住行中，“食”和人们的生活关系最密切。各地气候不同，生长的植物不同，做食物的材料当然也不同，风俗、习惯也大不一样。中国的南方产大米，所以南方人喜欢吃米饭。与此相反，北方产麦子，所以北方人喜欢吃饺子、面条。
下面写出该文本的难度等级(1-6级)：5级
文本：提起生物进化，人们多半会想到“物竞天择，适者生存”这 8 个字。在漫长的进化过程中，唯有战胜对手的幸运儿才能赢得大自然的青睐，拿到参加下一场物种角力的入场券。然而，大自然并不只是沿着单一的路线前行，“合则双赢，争则俱败”，体现互助与合作精神的共生或许是影响历史进程的另一重大因素。从表面上看，共生关系只是残酷竞争中的权宜之计，是特定条件下的巧合而已。然而生物学的研究却发现，这种生存战略同样是大自然的选择，是另一条进化道路——共生进化的产物。共生的形态多种多样，不拘一格。它存在于各层次、各种类生物的互动之中。海葵虾，顾名思义，对美丽的海葵情有独钟，它总是夹着海葵浪迹于海底世界。一遇危险，自有长着含毒触角的海葵出面摆平。这样一来，海葵虾可以放心觅食，不必为安全多费心机；而生性慵懒、喜静不喜动的海葵只要从共生伙伴的食物中分一杯羹就足以果腹。植物与动物共生的现象也不少见。生活在墨西哥的一种蚂蚁把巢筑在刺槐中空的树干中，享用刺槐叶柄分泌的富含糖分的汁液。作为回报，蚂蚁则负责植物的安全工作，一旦刺槐的敌人——食叶昆虫及其幼虫、草食动物靠近时，盛怒的蚁群就会蜂拥而出，与入侵者做殊死搏斗，直到把它们赶走。此外，蚂蚁还可以清除会对刺槐造成威胁的寄生植物。当这些植物靠近时，蚂蚁就会毫不客气地上前啃掉它们的藤条和嫩芽。高等植物与真菌的共生也早为人类所熟知。在这种共生关系中，真菌的菌丝体长在植物的根部，吸收植物光合作用的产物，而植物则可以从真菌的分解物中吸取养料。
下面写出该文本的难度等级(1-6级)：6级

文本如下：
[text]
下面写出该文本的难度等级(1-6级)：
'''


def read_hsk(path):
    dic_all = []
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        sentence, label = line.strip().split('\t')
        temp = {
            'text': sentence,
            'label': label
        }
        dic_all.append(temp)
    return dic_all


def test_hsk(input_path, out_path):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        dic_all = read_hsk(input_path)
        dic_all_new = []
        count = 0
        for dic in dic_all[:]:
            question = prompt_6shot.replace('[text]', dic['text'])
            res = llm_ask2(question)
            print(res)
            dic['response'] = res
            dic_all_new.append(dic)
            json.dump(dic, json_file, ensure_ascii=False)
            json_file.write('\n')
            count += 1


def clean_test(in_path, out_path):
    with open(in_path, 'r', encoding='utf-8') as json_file:
        lines = json_file.readlines()
    dic_all = []
    for line in lines:
        dic = json.loads(line)
        if dic['response']:
            match = re.search(r'为(\d+)级', dic['response'].replace(' ', ''))
            if match:
                difficulty_level = match.group(1)
            else:
                match = re.search(r'为：(\d+)', dic['response'].replace(' ', ''))
                if match:
                    difficulty_level = match.group(1)
                else:
                    match = re.search(r'(\d+)级', dic['response'].replace(' ', ''))
                    if match:
                        difficulty_level = match.group(1)
                    else:
                        difficulty_level = None
        else:
            difficulty_level = None
        dic['predict'] = difficulty_level
        dic_all.append(dic)
    with open(out_path, 'w', encoding='utf-8') as json_file:
        json.dump(dic_all, json_file, ensure_ascii=False, indent=2)


def llm_ask(question):
    client = OpenAI()
    model = 'gpt-3.5-turbo'
    response = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ],
        model=model,
        temperature=0
    )
    res = response.choices[0].message.content
    # print(res)
    return res


def llm_ask2(question):
    messages = [{'role': 'user', 'content': question}]
    model = 'qwen-14b-chat'
    # model = 'llama2-13b-chat-v2'
    # model = 'baichuan2-13b-chat-v1'
    # model = 'qwen-72b-chat'
    response = dashscope.Generation.call(
        model,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        # pprint(response)
        res = response.output.choices[0].message.content
        return res
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def llm_ask3(question):
    messages = [{'role': 'user', 'content': question}]
    gen = Generation()
    response = gen.call(
        'chatglm3-6b',
        messages=messages,
        result_format='message',  # set the result is message format.
    )
    res = response.output.choices[0].message.content
    return res


def evaluate_hsk(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        dataset = json.load(json_file)
    # print(dataset)
    labels = [int(data['label']) + 1 for data in dataset]
    predicts = [int(data['predict']) if data['predict'] else 6 for data in dataset]
    acc = metrics.accuracy_score(labels, predicts)
    report = metrics.classification_report(labels, predicts, digits=4)
    confusion = metrics.confusion_matrix(labels, predicts)
    print(acc)
    print(report)
    print(confusion)
    return acc, report, confusion


if __name__ == '__main__':
    input_path = 'test.txt'
    out_path = 'llm_result/gpt-3.5-turbo.json'
    pre_path = 'llm_result/gpt-3.5-turbo_cleaned.json'
    test_hsk(input_path, out_path)
    clean_test(out_path, pre_path)
    evaluate_hsk(pre_path)
