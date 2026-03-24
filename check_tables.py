import sqlite3

# This script helps to list down the tables available in mykart.db

conn = sqlite3.connect('mykart.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print("Tables in mykart.db:", tables)
conn.close()