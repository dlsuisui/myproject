# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:09:34 2020

@author: admin
"""

import pandas as pd
path = 'E:\\firesoon\\07_pooject\\associate\\'
def load_data(path):
    result = []
    with open(path) as f :
        for line in f :
            line = line.strip('\n')
            result.append(line.split(','))
    return result
#1.源数据
dataset = load_data(path+'data.csv')#list形式,每一行为list其中一个元素
#预览数据形式
for i in range(10):
    print(i+1,dataset[i],sep='->')  
"""
1->['shrimp', 'almonds', 'avocado', 'vegetables mix', 'green grapes', 'whole weat flour', 'yams', 'cottage cheese', 'energy drink', 'tomato juice', 'low fat yogurt', 'green tea', 'honey', 'salad', 'mineral water', 'salmon', 'antioxydant juice', 'frozen smoothie', 'spinach', 'olive oil']
2->['burgers', 'meatballs', 'eggs']
3->['chutney']
4->['turkey', 'avocado']
5->['mineral water', 'milk', 'energy bar', 'whole wheat rice', 'green tea']
6->['low fat yogurt']
7->['whole wheat pasta', 'french fries']
8->['soup', 'light cream', 'shallot']
9->['frozen vegetables', 'spaghetti', 'green tea']
10->['french fries']
"""
#2.文本数据映射到数字，提高运算性能
import itertools
items = set(itertools.chain(*dataset)) #提取所有的项目
#用来保存字符串到编号的映射
str_to_index = {}
#用来保存编码到字符串的映射
index_to_str = {}
for index,item in enumerate(items):
    str_to_index[item] = index
    index_to_str[index] = item
#输出结果
print('字符串到编号:',list(str_to_index.items())[:5])
print('编码到字符串:',list(index_to_str.items())[:5])  

"""
字符串到编号: [('sparkling water', 0), ('fromage blanc', 1), ('mineral water', 2), ('cake', 3), ('tea', 4)]
编码到字符串: [(0, 'sparkling water'), (1, 'fromage blanc'), (2, 'mineral water'), (3, 'cake'), (4, 'tea')]
"""  
#将原始字符串数据转换为数值索引
for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        dataset[i][j] = str_to_index[dataset[i][j]]
for i in range(10):
    print(i+1,dataset[i],sep='->')
"""
1->[100, 66, 20, 21, 104, 75, 17, 22, 50, 9, 63, 33, 72, 7, 2, 47, 107, 8, 64, 89]
2->[52, 105, 14]
3->[38]
4->[68, 20]
5->[2, 114, 25, 108, 33]
6->[63]
7->[44, 45]
8->[10, 62, 86]
9->[53, 113, 33]
10->[45]

"""
#3.生成关联关系
#3.1生成候选1项集
def buildC1(dataset):
    """
    创建病返回频繁1项集列表
    Parameters
    ----------
    dataset：list 数据集对象
    Returns
    ----------
    c1：list 候选1项集列表
    """
    item1 = set(itertools.chain(*dataset))
    return [frozenset([i]) for i in item1]

c1 = buildC1(dataset)

#3.2根据候选1项集生成频繁项集
def ck_to_lk(dataset,ck,min_support):
    """
    根据候选k项集与对应的最小支持度，创建生成频繁k项集
    Parameters
    ----------
    dataset：list 数据集对象
    ck：set/list 所有候选k项集构成的set/list
    min_supprt:float 最小支持度 候选k项集的支持度大于等于最小支持度,则候选k项集就是频繁k项集
    Retruns
    ----------
    Lk:dict 所有频繁k项集构成的字典。字典中可以为频繁k项集，字典values为频繁k项集对应的支持度
    """
    support={}
    for row in dataset:
        for item in ck:
            #判断项集是否在记录行中出现
            if item.issubset(row):
                support[item]=support.get(item,0)+1
    total = len(dataset)
    return {k: v/total for k,v in support.items() if v/total >= min_support}
L1 = ck_to_lk(dataset,c1,0.05)
L1
#3.2 频繁k项集组合生成候选k+1项集
def lk_to_ck(lk_list):
    """
    根据参数指定的频繁k项集，组合生成候选k+1项集
    Parameters
    ----------
    lk_list: list 所有频繁k项集构成的列表
    Returns
    ----------
    返回所有候选k+1项集构成的set
    """
    #保存所有组合之后的候选k+1项集
    ck = set()
    lk_size = len(lk_list)
    #如果频繁项集的是数量小于1，则不可能再通过组合生成候选k+1项集
    #直接返回空set即可
    if lk_size>1:
        #获取频繁k项集的k值
        k = len(lk_list[0])
        #h获取任意两个元素序号的索引组合
        for i,j in itertools.combinations(range(lk_size),2):
            #将对应未知的两个频繁k项集进行组合,生成一个新的项集
            t = lk_list[i]|lk_list[j]
            #如果组合之后的项集是k+1项集，则为候选k+1项集，加入到结果set中
            if len(t) == k+1:
                ck.add(t)
    return ck

c2 = lk_to_ck(list(L1.keys()))
c2

#3.3生成所有频繁项集
def get_L_all(dataset,min_support):
    """
    根据最小支持度，从指定的数据集中，返回所有的频繁项集
    Parameters
    ----------
    dataset:list 数据集列表
    min_support: float 最小支持度
    Returns
    ----------
    L_all：dict 所有频繁项集构成的字典
    
    """  
    c1 = buildC1(dataset)
    L1 = ck_to_lk(dataset,c1,min_support)
    #定义字典，保存所有的频繁k项集
    L_all = L1
    Lk = L1
    #当频繁项集中的元素(键值对)数量大于1时，才有可能
    #组合生成候选k+1项集
    while len(Lk)>1:
        lk_key_list = list(Lk.keys())
        #由频繁项集生成候选k+1项集
        ck = lk_to_ck(lk_key_list)
        #由候选k+1项集组合生成频繁k+1项集
        Lk = ck_to_lk(dataset,ck,min_support)
        #如果频繁k+1项集字典不为空，则将所有频繁k+1项集加入到L_all字典中
        if len(Lk)>0:
            L_all.update(Lk)
        else:
            #否则，频繁k+1项集为空，退出循环
            break
    return L_all

L_all = get_L_all(dataset, 0.05)

#3.4生成关联规则
##从一个频繁项集中生成关联规则
def rules_from_item(item):
    """
    从参数指定的频繁项集中生成关联规则
    Parameters
    ----------
    item : frozenset 频繁项集
    Returns
    ----------
    rules：tuple of list
    频繁项集对应的所有关联规则组合
    """
    #定义规则左侧的列表
    left = []
    for i in range(1,len(item)):
        left.extend(itertools.combinations(item,i))
    return [(frozenset(le),frozenset(item.difference(le))) for le in left]

rules_from_item(frozenset({1,2,3}))
##生成所有频繁k项集的字典后，遍历字典中每一个键，进而计算该频繁项集所有可能的关联规则，然后对每一关联规则，计算置信度，保留符合最小置信度的关联规则
def rules_from_L_all(L_all,min_confidence):
    """
    从所有频繁项集字典中，生成关联规则列表
    Parameters
    ----------
    L_all:dict 所有频繁项集
    min_confidence:float 最小置信度
    Returns
    ----------
    result:dict of list 所有满足最小置信度的关联规则
    
    """
    #保存所有候选的关联规则
    rules = []
    for Lk in L_all:
        #如果频繁项集的元素个数为1，则无法生成关联规则，不予考虑
        if len(Lk)>1:
            rules.extend(rules_from_item(Lk))
    result = []
    for left,right in rules:
        support = L_all[left | right]
        confidence = support / L_all[left]
        lift = confidence / L_all[right]
        if confidence >= min_confidence:
            result.append({'左侧':left,'右侧':right,'支持度':support,'置信度':confidence,"提升度":lift})
    return result

rules_from_L_all(L_all,0.3)


#3.5最终程序
def apriori(dataset,min_support,min_confidence):
    """
    从参数指定的数据集中，根据最小支持度和最小置信度，返回所有满足条件的关联规则
    Parameters
    ----------
    dataset:list 数据集
    min_support :float 最小支持度
    min_confidence:float 最小置信度
    Returns
    ----------
    result：dict of list 所有满足条件的关联规则
    
    """
    L_all= get_L_all(dataset,min_support)
    rules = rules_from_L_all(L_all,min_confidence)
    return rules
rules = apriori(dataset,0.05,0.3)
rules
#使用dataset展示结果
def change(item):
    """
    从参数指定的数据集中，根据最小支持度与最小置信度，返回所有满足条件的关联规则
    Parameters
    ----------
    item :frozenset 关联规则中，左侧或右侧的部分
    Returns
    ----------
    result：list 由index转换为文本后的内容
    
    """
    li = list(item)
    for i in range(len(li)):
        li[i] = index_to_str[li[i]]
    return li

df = pd.DataFrame(rules)
df = df.reindex(['左侧','右侧','支持度','置信度','提升度'],axis=1)
df['左侧']=df['左侧'].apply(change)
df['右侧']=df['右侧'].apply(change)
df













        
