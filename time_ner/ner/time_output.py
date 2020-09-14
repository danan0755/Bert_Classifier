#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author ChenYongSheng
# date 20200831
import json
import time
import re
from datetime import datetime

import arrow

from ner.time_normalizer import TimeNormalizer

"""时间识别接口"""


# 时间实体识别接口  reg用来控制用户第一句话说的是早上，下午，晚上，默认为0，早上=-2，中午=-3，下午晚上=-4
def time_output(query, userid, tagid, ner_date, ner_time, reg, msgIdServer):
    start = time.time()

    # 获取当前时间
    now = int(time.time())
    timeArray = time.localtime(now)
    now = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

    tn = TimeNormalizer()
    timeBase = arrow.now()
    reg_send = 0
    t_date = ner_date
    t_time = ner_time

    res = tn.parse(target=query, timeBase=timeBase)
    res = json.loads(res)

    rule = u"凌晨|早上|早晨|早间|晨间|今早|明早|早|清晨|上午|中午|午间"
    pattern = re.compile(rule)
    match = pattern.search(query)

    now_date = datetime.now().strftime('%Y-%m-%d')

    if 'timestamp' in res:
        timestamp = res['timestamp']
        sp = str(timestamp).split(' ')
        # 包含日期和时间
        if len(sp) > 1:
            t_date = sp[0]
            t_time = sp[1]
            # 判断是否需要更新日期
            if ner_date:
                if ner_date > t_date and t_date == now_date:
                    t_date = ner_date
                else:
                    t_date = t_date
            t_date, t_time = check_time(res, t_date, t_time, now, match, reg)
        elif 'timetype' in res:
            # 只有日期，没有时间
            t_date = sp[0]
            reg_send = res['timetype']
    elif 'timespan' in res:
        timespan = res['timespan'][0]
        sp = str(timespan).split(' ')
        if len(sp) > 1:
            t_date = sp[0]
            t_time = sp[1]
            # 判断是否需要更新日期
            if ner_date:
                if ner_date > t_date and t_date == now_date:
                    t_date = ner_date
                else:
                    t_date = t_date
            t_date, t_time = check_time(res, t_date, t_time, now, match, reg)
        elif 'timetype' in res:
            t_date = sp[0]
            reg_send = res['timetype']
    else:
        return 'null'

    # 根据缓存的信息，判断时间是否需要加12小时
    if t_time and reg == -4 and not match:
        hour, minute, second = split_time(t_time)
        hour = int(hour) + 12
        if hour > 24:
            t_time = str(hour - 12) + ":" + minute + ":" + second
        else:
            t_time = str(hour) + ":" + minute + ":" + second

    t_date_time = t_date + ' ' + t_time
    if t_date_time < now:
        t_time = ''
    if t_date or t_time:
        end = time.time()
        consume = int((end - start) * 1000)
        output = {
            "query": query,
            "tagid": tagid,
            "userid": userid,
            "ner_date": t_date,
            "ner_time": t_time,
            "msgIdServer": msgIdServer,
            "reg": reg_send,
            "time": consume,
            "result": 1
        }
    else:
        output = 'null'
    return output


# 时间切割
def split_time(time):
    tmp_time = str(time).split(":")
    hour = tmp_time[0]
    minute = tmp_time[1]
    second = tmp_time[2]
    return hour, minute, second


# 日期切割
def split_date(date):
    tmp_date = str(date).split("-")
    year = tmp_date[0]
    month = tmp_date[1]
    day = tmp_date[2]
    return year, month, day


''' 1、对于用户说：今天8点 ，应识别出08:00和20:00，取值逻辑如下：

    如当前时间为7点，小于用户所说的8点，则取值08:00；

    如当前时间为15点，大于用户所说的8点，小于8+12点，则取值20:00；

    如当前时间为21点，大于用户所说的8点，大于8+12点，视用户所说时间已过期，不做取值；'''

'''2、用户仅说了日期，只返回日期；用户仅说了时间，则默认为当天，对于用户说0点和24点，默认取0点，时间加一天。在time_unit.py处理了24点为0点'''

'''3、当用户说了时段&24小时制，如早上18:00，按照24小时制取值即可，应取18:00；在time_normalizer.py处理了'''

'''4、对于用户说早上8点，不用进行第一条判断，第一条判断条件是用户说8点或者今天8点'''


def check_time(res, t_date, t_time, now, match, reg):
    if 'timestamp' in res:
        res = res['timestamp']
    elif 'timespan' in res:
        res = res['timespan'][0]
    reg_hour, reg_minute, reg_second = split_time(t_time)

    now_date = str(now).split(" ")[0]
    now_time = str(now).split(" ")[1]
    now_hour, _, _ = split_time(now_time)
    if reg_hour == '00':
        t_time = t_time
        # 如果当前日期和识别日期为同一天，则天数加一
        if t_date == now_date:
            t_date = add_day(t_date)
    elif now <= res:
        t_time = t_time
    elif now > res and not match and reg != -2 and t_date == now_date:
        reg_hour = int(reg_hour) + 12
        if reg_hour > 24:
            t_time = ''
        elif reg_hour == 24:
            t_time = '00:00:00'
            # 如果当前日期和识别日期为同一天，则天数加一
            if t_date == now_date:
                t_date = add_day(t_date)
        elif int(now_hour) > reg_hour:
            t_time = ''
        else:
            t_time = str(reg_hour) + ":" + reg_minute + ":" + reg_second
    # 校验时间是否过期
    t = t_date + " " + t_time
    if t < now:
        t_time = ''
    return t_date, t_time


def add_day(now):
    date = str(now).split(" ")[0]
    year, month, day = split_date(date)
    if day.startswith('0'):
        day = int(day)
        day += 1
        if len(str(day)) < 2:
            day = "0" + str(day)
    else:
        day = int(day)
        day += 1
        day = str(day)
    t_date = year + "-" + month + "-" + str(day)
    return t_date


if __name__ == '__main__':
    # 输入小时数小于等于当前小时数，日期自动加一天，如现在是2020-08-04 10点，输入“10点”，输出结果为2020-08-05 10：00：00
    query = '今天早上8点'
    res = time_output(query, None, None, ner_date='2020-09-10', ner_time=None, reg=0, msgIdServer=1)
    print(res)
