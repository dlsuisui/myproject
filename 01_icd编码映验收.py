# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:43:14 2020

@author: admin
"""

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import Levenshtein
import sys
path = 'E:\\firesoon\\07_pooject\\icdMapping\\'

def fixCN(word):
    return word.lower().replace('，','').replace('的','')

def console(log):
    print(log)
    sys.stdout.flush()

def Score(name,code):
    """
    同时考虑icd编码和名称的相似度；
    名称的相似度大于95%的话，赋予一个较高10的权重；
    编码的相似度=1的话，赋予一个相对较高2的权重；
    其他情况下直接对编码名称进行组合得到一个综合权重
    """
    score=0
    if name > 0.95:
        score=10
    elif code ==1:
        score=2
    else:
        score=0.8*code+name
    return score


#1.需要映射的诊断编码和人工映射表
hzyy_diag_lack = pd.read_csv(path+'hzyy_diag_lack.csv',dtype={'diag_code':str,'diag_name':str}).rename(columns={'diag_code':'icd_code','diag_name':'icd_name'})
diag_mapping_to_ins = pd.read_excel(path+'diag_mapping_to_ins.xlsx',dtype={'diag_code':str,'diag_name':str}).rename(columns={'diag_code':'icd_code','diag_name':'icd_name','diag_code_yb':'icd_code_yb','diag_name_yb':'icd_name_yb'}).drop_duplicates(subset=['icd_code','icd_name'],keep='first')
res1 = pd.merge(hzyy_diag_lack, diag_mapping_to_ins,on=['icd_code','icd_name'],how='left')
res1_remain = res1[~(res1['icd_code_yb'].isnull())][['icd_code','icd_name','icd_code_yb','icd_name_yb']]
res1_remain.loc[:,'mark']='精确匹配'

#2.目标版本直接根据名称匹配
yb_diag = pd.read_csv(path+'diag_version_yb.csv',dtype={'diag_code':str,'diag_code':str},usecols=['diag_code','diag_name','version']).rename(columns={'diag_code':'icd_code','diag_name':'icd_name'})
yb_diag = yb_diag[yb_diag['version']=="《医保版20191201》"]
del yb_diag['version']
yb_diag['icd_name'] = yb_diag['icd_name'].apply(fixCN)
yb_diag.rename(columns={'icd_code':'icd_code_yb','icd_name':'icd_name_yb'},inplace=True)

res1_unmath = res1[res1['icd_code_yb'].isnull()][['icd_code','icd_name']]
res2 = pd.merge(res1_unmath, yb_diag,left_on='icd_name',right_on='icd_name_yb',suffixes=('_raw','_yb'),how='left')
res2_remain = res2[~(res2['icd_code_yb'].isnull())]
res2_remain.loc[:,'mark']='精确匹配'
res2_unmatch = res2[res2['icd_code_yb'].isnull()][['icd_code','icd_name']]

#3.剩余未匹配的直接通过算法进行模糊匹配

count = 0
s = len(res2_unmatch)
if s==0:
    console('本数据全为精确匹配！')
    res_final = pd.concat([res1_remain,res2_remain],axis=0).reset_index(drop=True)
    res_final.to_csv(path+'output_data.csv',index=False) #直接输出精确匹配结果
else:
    console('剩余%d条编码需要模糊匹配'%s) #打印需要进行模糊匹配的数量
    # 使用分词后的名称进行模糊匹配
    df = pd.DataFrame(columns=['icd_code_yb','name_yb'])
    console("开始模糊匹配")
    for item in list(zip(res2_unmatch['icd_code'],res2_unmatch['icd_name'])):
        count +=1
        console("模糊匹配进度: 第%d编码进行正在进行匹配" % count)
        yb_diag['codeScore'] = yb_diag['icd_code_yb'].apply(lambda x: Levenshtein.jaro_winkler(item[0][:-1],x)) #对于icd编码使用Levenshtein.jaro_winkler
        # 忽略顺序匹配
        yb_diag['nameScore'] = yb_diag['icd_name_yb'].apply(lambda x: fuzz.token_sort_ratio(item[1],x)/100)#对于icd编码使用fuzz.token_sort_ratio

        yb_diag['finalScore'] = yb_diag[['codeScore','nameScore']].apply(lambda x:Score(x['nameScore'],x['codeScore']),axis=1)#根据编码和名称的相似度得到最终的相似度

        df1 = yb_diag.iloc[yb_diag.finalScore.argmax(),:]#[['icd_code_yb','icd_name_yb']]
        df = df.append(df1)
        if count %10==0:
            _ = count//10*10
            console('总计 %d 条数据需要进行模糊匹配，已匹配 %d 条数据! '%(s,_) )
        if count==s:
            console('全部数据匹配完成！')

    res2_unmatch.index=range(len(res2_unmatch))
    df.index=range(len(df))
    res3_remain = pd.merge(res2_unmatch[['icd_code','icd_name']],df[['icd_code_yb','icd_name_yb']],left_index=True,right_index=True,how='inner')
    res3_remain.loc[:,'mark']= res3_remain.apply(lambda x:'精确匹配' if x.icd_name==x.icd_name_yb else '模糊匹配',axis=1)
    
    res_final = pd.concat([res1_remain,res2_remain,res3_remain],ignore_index=True)
    res_final = res_final[['icd_code','icd_name','icd_code_yb','icd_name_yb','mark']]
    res_final.to_csv(path+'output_data.csv',index=False)
        