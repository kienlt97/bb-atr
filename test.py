
from datetime import datetime
if __name__ == '__main__':
    str_date = datetime.fromtimestamp(float('1715170200772') / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    print(str_date)
