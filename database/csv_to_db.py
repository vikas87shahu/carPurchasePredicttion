import sqlite3
conn = sqlite3.connect('car_prediction.db')
cursor = conn.cursor()

# RUN ONCE STUFF
cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Age INTEGER,
        EstimatedSalary INTEGER,
        Purchased INTEGER
    )
''')
conn.commit()

"""## Importing csv to database (One time Process)"""

#Importing csv to db
import csv
with open('database/Social_Network_Ads.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row if it exists
    for row in csv_reader:
        age = int(row[0])
        salary = int(row[1])
        purchased = int(row[2])
        cursor.execute('INSERT INTO customers (Age, EstimatedSalary, Purchased) VALUES (?, ?, ?)', (age, salary, purchased))
conn.commit()