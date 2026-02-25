import csv
with open('datasets/recruitment.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if len(row) != 11:
            print(f'Line {i+1}: {len(row)} fields')
            break