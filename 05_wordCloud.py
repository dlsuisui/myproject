# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:51:02 2020

@author: admin
"""
import pandas as pd
import jieba
path='E:\\firesoon\\07_pooject\\newsplit\\dat\\'
#1.源数据
df_news = pd.read_table(path+'val.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()
df_news.columns
"""
'category':新闻类别
'theme':主题
'URL':网址
'content':新闻主题内容
"""
#2.分解
content = df_news.content.values.tolist()  # 转换为list 实际上是二维listcontent
#print(content[1000])
content_S=[]
for line in content:
    current_segment = jieba.lcut(line)
    #\r 回车换行; \n 换行 列表不存在为空且不是换行
    if len(current_segment)>1 and current_segment!='\r\n': 
        content_S.append(current_segment)
#转换为dataframe
df_content = pd.DataFrame({'content_S':content_S})   # 转换为DataFrame
    

#3.去除一些停用词
# 读取停词表
stopwords = pd.read_csv(path+'stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
stopwords.head() 
   
# 删除新闻中的停用词 --即从content中去除一些停用词
def drop_stopwords(contents, stopwords):
    contents_clean = [] # 删除后的新闻
    all_words = []  # 构造词云所用的数据
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words
  
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()

# 得到删除停用词后的新闻以及词云数据
contents_clean, all_words = drop_stopwords(contents, stopwords) 
 
# df_content.content_S.isin(stopwords.stopword)
# df_content=df_content[~df_content.content_S.isin(stopwords.stopword)]
# df_content.head()   

#查看删除停用词之后的内容
df_content = pd.DataFrame({'contents_clean':contents_clean})
df_content.head()

#查看删除停用词之后的词
df_all_words = pd.DataFrame({'all_words':all_words})
df_all_words.head()

#4.进行词云展示
# 分组统计
import numpy
words_count = df_all_words.all_words.value_counts().reset_index().rename(columns={'index':'all_words','all_words':'count'})
words_count = words_count.sort_values(by=['count'],ascending=False)
words_count.head()
#词云展示
from wordcloud import WordCloud # 词云库
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0,5.0)
wordcloud = WordCloud(font_path='./data/simhei.ttf',background_color='white',max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values} # 这里只显示词频前100的词汇
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
