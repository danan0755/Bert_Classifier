时间实体识别

1、只输入日期，则只返回日期，不返回时间

2、只输入时间，默认日期为今天

3、由于业务的需要，对于用户输入“早上，中午，下午晚上”等不包含具体时间的，返回timetype，对应-2，-3，-4

tn = TimeNormalizer() timeBase = arrow.now()

res = tn.parse(target=u'2013年二月二十八日下午四点三十分二十九秒', timeBase=timeBase) # target为待分析语句，timeBase为基准时间默认是当前时间 print(res) #{"type": "timespan", "timetype": 0, "timespan": ["2013-02-28 16:30:29"]}

res = tn.parse(target=u'下午3点到5点') print(res) #{"type": "timespan", "timespan": ["2020-09-05 15:00:00", "2020-09-05 17:00:00"]}

res = tn.parse(target='早上', timeBase=timeBase) print(res) #{"type": "timestamp", "timestamp": "2020-09-05", "timetype": -2}

res = tn.parse(target='今天0点', timeBase=timeBase) print(res) #{"type": "timestamp", "timestamp": "2020-09-05 00:00:00"}

res = tn.parse(target='今天24点', timeBase=timeBase) print(res) #{"type": "timestamp", "timestamp": "2020-09-06 00:00:00"}
