# -*- coding: utf-8 -*-
import datetime
import json
import re
import time
import arrow

from ner.time_normalizer import TimeNormalizer

import sys

sys.path.append('.')

tn = TimeNormalizer()
timeBase = arrow.now()

#
res = tn.parse(target='今天')  # target为待分析语句，timeBase为基准时间默认是当前时间
print(res)


res = tn.parse(target='8月22')
print(res)


res = tn.parse(target=u'2013年二月二十八日下午四点三十分二十九秒', timeBase=timeBase)  # target为待分析语句，timeBase为基准时间默认是当前时间
print(res)

res = tn.parse(target=u'下周一去怀北滑雪场，上午8点半在图书馆前集合')  # target为待分析语句，timeBase为基准时间默认是当前时间
print(res)

res = tn.parse(target=u'今年儿童节晚上九点一刻')  # target为待分析语句，timeBase为基准时间默认是当前时间
print(res)

res = tn.parse(target=u'三日')  # target为待分析语句，timeBase为基准时间默认是当前时间
print(res)

res = tn.parse(target=u'7点4')  # target为待分析语句，timeBase为基准时间默认是当前时间
print(res)

res = tn.parse(target=u'今年春分')
print(res)

res = tn.parse(target=u'这周六上午八点半去怀柔下午5:30回学校')
print(res)

res = tn.parse(target=u'下午3点到5点')
print(res)

res = tn.parse(target=u'今天早上')
print(res)

res = tn.parse(target=u'下午')
print(res)

res = tn.parse(target=u'9月1号上午')
print(res)

res = tn.parse(target='我一般下午4-6点有时间看房')
print(res)

res = tn.parse(target='我一般下午4到6点有时间看房')
print(res)

res = tn.parse(target='9月22号晚上7点', timeBase=timeBase)
print(res)


res = tn.parse(target='8月22日中午12点', timeBase=timeBase)
print(res)

res = tn.parse(target='请问今晚大概6点可以带看房子吗', timeBase=timeBase)
print(res)

res = tn.parse(target='早上', timeBase=timeBase)
print(res)

res = tn.parse(target='今天8点', timeBase=timeBase)
print(res)


res = tn.parse(target='今天早上18点', timeBase=timeBase)
print(res)

res = tn.parse(target='今天0点', timeBase=timeBase)
print(res)

res = tn.parse(target='今天24点', timeBase=timeBase)
print(res)

res = tn.parse(target='明天0点', timeBase=timeBase)
print(res)

res = tn.parse(target='明天24点', timeBase=timeBase)
print(res)

res = tn.parse(target='下午8点', timeBase=timeBase)
print(res)

res = tn.parse(target='明天晚上12点', timeBase=timeBase)
print(res)


res = tn.parse(target='8月22下午1点', timeBase=timeBase)
print(res)


