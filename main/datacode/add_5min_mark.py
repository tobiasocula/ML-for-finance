import os

for filename in os.listdir('../data/stocks_5m'):
    print('filename:', filename)
    new = filename.split('.')[0] + '_5m.csv'
    os.rename('../data/stocks_5m/' + filename, '../data/stocks_5m/' + new)