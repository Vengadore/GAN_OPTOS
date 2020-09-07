import time, sys
from IPython.display import clear_output

class Progress_bar():
    def __init__(self,num_elements,bar_length = 20):
        self.bar_length = bar_length
        self.element = -1
        self.total_elements = num_elements
        self.update_progress()
    def update_progress(self):
        self.element = self.element+1
        progress = self.element/self.total_elements
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        clear_output(wait = True)
        block = int(round(self.bar_length * progress))
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (self.bar_length - block), progress * 100)
        print(text)
