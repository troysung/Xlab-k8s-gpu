import pandas as pd
csv=pd.read_csv('./kddcup.newtestdata_10_percent_unlabeled.csv',low_memory=False,error_bad_lines=False)#读取csv中的数据
df = pd.DataFrame(csv)
print(df.shape)#打印行数
f=df.drop_duplicates(keep=False)#去重
print(f.shape)#打印去重后的行数
f.to_csv('./duplicated kddcup.newtestdata_10_percent_unlabeled.csv',index=None)#写到一个新的文件
