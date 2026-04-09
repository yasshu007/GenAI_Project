import sqlite3
import pandas as pd


df = pd.read_excel('sales.xlsx')

try:
    with sqlite3.connect("mykart.db") as conn:
        print(f"SQLite connection with {sqlite3.sqlite_version} successfully.")
        
        df.to_sql('sales_data', conn, if_exists='replace', index=False)
        results = pd.read_sql('SELECT * FROM sales_data LIMIT 15', conn)
        print(results)

    conn.close()
except sqlite3.OperationalError as e:
    print("Failed to open SQLite:", e)


