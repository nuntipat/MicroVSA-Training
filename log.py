import sys

def show_log(str, file=None):
    if file is not None:
        print(str, file=file)