import numpy as np
import jieba
import os
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 统计所有的标点符号和英文字符
punctuation = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！？｡。＂" \
           "＃＄％＆＇()＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝‘’" \
           "～｟｠｢｣､、〃《》「」『』【】〔〕（）〖〗〘〙〚〛〜〝〞“”〟〰\n\u3000" \
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" \
              "说道一个自己咱们什么不是他们一声心想心中知道只见还是却是甚么突然 "


def remove_punctuation(something):  # 移除列表中的标点符号
    new_l = []
    for s in something:
        if s not in punctuation:
            new_l.append(s)
    return new_l


def book_read():  # 将所有的book读取，使用jieba进行分词，再去除其中的标点
    if os.path.exists('books.npy'):
        a = np.load('books.npy', allow_pickle=True)
        books = a.tolist()
    else:
        books_name = open('source/inf.txt').read()
        book_list = books_name.split(",")

        books = []
        i = 0

        for book in book_list:
            i += 1
            f = open('source/'+book+'.txt', encoding="UTF-8")
            txt = f.read()
            # seg_list = jieba.lcut(txt, cut_all=False)
            seg_list = [w for w in jieba.cut(txt) if len(w) > 1]
            seg = remove_punctuation(seg_list)
            books.append(seg)
            print('读取进度{}/16'.format(i))
        m = np.array(books, dtype=object)
        np.save('books.npy', m)
    return books


# books = book_read()  # 读取所有书
books = np.load('books.npy', allow_pickle=True)
books = books[1]
model = Word2Vec(books, hs=1, min_count=10, window=5, vector_size=200, sg=1, epochs=200)
print(model)
result = model.wv.similar_by_word(books[20])
print(books[20])
print(result)


# testword = ['张无忌', '乔峰', '郭靖', '杨过', '令狐冲', '韦小宝']
# for i in range(len(testword)):
#     result = model.wv.most_similar(testword[i])
#     print(testword[i])
#     print(result)
