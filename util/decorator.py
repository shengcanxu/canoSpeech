
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 在这里处理异常
            print(f"捕获到异常：{e}")
            # 可以返回一个默认值或者重新抛出异常
            return None
    return wrapper