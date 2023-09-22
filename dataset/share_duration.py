class ShareDuration:
    def __init__(self):
        self.data = {}

    def get(self, filename):
        # print(filename)
        data = self.data.get(filename, None)
        if data is None:
            # print(len(self.data))
            # print(None)
            return None
        else:
            data["read_count"] += 1
            print(data)
            return data["value"]

    def set(self, filename, value):
        self.data[filename] = {"value": value, "read_count": 0}
