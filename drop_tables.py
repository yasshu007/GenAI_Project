import sqlite3

# This script helps to delete tables required in mykart.db

conn = sqlite3.connect('mykart.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print("Tables in mykart.db:", tables)
print("="* 30)
# This will delete the table provided if available in the database: mykart.db
cursor.execute("DROP TABLE mykart;")        # Table needs to be provided here.
conn.close()

