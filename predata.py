#
import re

import jieba
from sklearn.feature_extraction.text import CountVectorizer

class analyspromption:
    def __init__(self):
        pass

    #进行数据清洗
    def clean_text(self,text_1):
        pro_clean = []
        for text in text_1:
            text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', ' ', text)
            text = re.sub(r'[1234567890]', ' ', text)
            if text == None:
                print("error:清洗数据出现错误")
            pro_clean.append(text)
        return pro_clean

    #进行分词
    def fenci(self,text_2):
        pro_fenci = []
        for text in text_2:
            demo = jieba.lcut(text)
            pro_fenci.append(demo)

        return pro_fenci

    #读取文件中的标记数据
    #我的文本中最后一个是标识符。
    def load_text(self,path,encoding = 'utf-8'):
        text = []
        label = []
        with open(path,'r',encoding=encoding) as f:
            for line in f:
                line = line.rstrip('\n\r')   #去掉每行末尾的换行符
                left, right = re.split(r'\s+', line.strip(), maxsplit=1)
                right = right.strip()
                if right not in ('0','1'):
                    continue

                text.append(left)
                label.append(right)
        return text,label



    #使用n-gram进行特征提取
    def tezheng(self, texts, n):
        if not isinstance(texts, list):
            texts = [texts]

        # 确保输入是空格分隔的字符串
        processed_texts = []
        for text in texts:
            if isinstance(text, list):
                # 如果是分词列表，用空格连接
                processed_texts.append(' '.join(text))
            elif isinstance(text, str):
                processed_texts.append(text)
            else:
                processed_texts.append(str(text))
        # 使用CountVectorizer提取N-gram特征

        vectorizer = CountVectorizer(
            ngram_range=(1, n),
            analyzer='word',
            min_df=1,
            token_pattern=r'(?u)\b\w+\b'
        )

        vector = vectorizer.fit_transform(processed_texts)

        return vector, vectorizer






