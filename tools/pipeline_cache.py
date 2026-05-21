class PipelineCache:
    def __init__(self):
        self.cache = {}

    def make_key(self, dataset_name, actions):
        return dataset_name + "::" + "->".join(actions)

    def has(self, dataset_name, actions):
        key = self.make_key(dataset_name, actions)
        return key in self.cache

    def get(self, dataset_name, actions):
        key = self.make_key(dataset_name, actions)
        return self.cache[key]

    def set(self, dataset_name, actions, result):
        key = self.make_key(dataset_name, actions)
        self.cache[key] = result

    def size(self):
        return len(self.cache)
