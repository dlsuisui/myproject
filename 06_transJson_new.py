# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:44:11 2019

@author: admin
"""
# =============================================================================
# #1.针对RESPONSE_JSON处理
# =============================================================================
import pandas as pd
import json
path = 'E:\\firesoon\\00_旧\\分组器\\DRG_class\\地区优化\\佛山\\新JSON参数转化\\dat\\'
def transResponse(x):
    return x[1:len(str(x))-1]

#1.针对REQUEST_JSON进行替换
#基础数据处理
df_jx = pd.read_csv(path+'佛山市一2020-07分组参数结果Guangdong_Foushan_202007_20200821.csv')
#分组器请求参数处理
df_jx_temp= df_jx[['PID','RESPONSE_JSON']]
df_jx_temp.RESPONSE_JSON = df_jx_temp.RESPONSE_JSON.map(lambda x:transResponse(x))

df_jx_temp_dcit = df_jx_temp.to_dict(orient='records')
for i in df_jx_temp_dcit:
    i.update(json.loads(i['RESPONSE_JSON']))
    del i['RESPONSE_JSON']
df_reqjson = pd.DataFrame(df_jx_temp_dcit)

# =============================================================================
# #1.针对REQUEST_JSON处理
# =============================================================================
import pandas as pd
import json
path = 'E:\\firesoon\\00_旧\\分组器\\DRG_class\\地区优化\\佛山\\新JSON参数转化\\dat\\'
#1.针对REQUEST_JSON进行替换
#基础数据处理
df_jx = pd.read_excel(path+'佛山地区分组参数.xlsx',sheet_name='Sheet1')
#分组器请求参数处理
df_jx_temp= df_jx[['PID','REQUEST_JSON']]

df_jx_temp_dcit = df_jx_temp.to_dict(orient='records')
for i in df_jx_temp_dcit:
    i.update(json.loads(i['REQUEST_JSON']))
    del i['REQUEST_JSON']
df_reqjson = pd.DataFrame(df_jx_temp_dcit)



