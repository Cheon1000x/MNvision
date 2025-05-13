import datetime

def current_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
