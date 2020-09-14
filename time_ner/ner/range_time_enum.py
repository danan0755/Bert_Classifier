
# -*- coding: utf-8 -*-


# 范围时间的默认时间点
class RangeTimeEnum():
    # day_break = 3  # 凌晨
    # early_morning = 8  # 早
    # morning = 10  # 上午
    # noon = 12  # 中午、午间
    # afternoon = 15  # 下午、午后
    # night = 18  # 晚上、傍晚
    # lateNight = 20  # 晚、晚间
    # midNight = 23  # 深夜

    day_break = 6  # 凌晨
    early_morning = 11  # 早
    morning = 11  # 上午
    noon = 12  # 中午、午间
    afternoon = 23  # 下午、午后
    night = 23  # 晚上、傍晚
    lateNight = 23 # 晚、晚间
    midNight = 23  # 深夜


if __name__ == "__main__":
    print(RangeTimeEnum.afternoon)
