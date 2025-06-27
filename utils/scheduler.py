class LinearScheduler:
    """
    线性衰减调度器。
    在指定的步数内，将一个值从初始值线性地衰减到最终值。

    :param start_value: 初始值
    :param end_value: 最终值
    :param duration: 衰减过程持续的总步数
    """
    def __init__(self, start_value: float, end_value: float, duration: int):
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

    def value(self, step: int) -> float:
        """
        根据当前步数计算参数的当前值。

        :param step: 当前的训练步数
        :return: 计算出的参数值
        """
        # 计算衰减的完成度（0.0 到 1.0 之间）
        fraction = min(float(step) / self.duration, 1.0)
        return self.start_value + fraction * (self.end_value - self.start_value)