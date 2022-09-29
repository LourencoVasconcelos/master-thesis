from itertools import count


class SeedGenerator:
    def __init__(self, start=0, step=1):
        self.generator = count(start=start, step=step)

    def next(self):
        return next(self.generator)
