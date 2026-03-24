import sqlite3
import pandas as pd

def extract_and_prepare_data(db_path):
    conn = sqlite3.connect(db_path)
    
    # We combine relevant categorical fields into a 'context' string
    query = """
    SELECT 
    Cust_Id,
    -- Constructing a descriptive sentence for the Vector Store
    'Customer ' || COALESCE(Cust_Name, 'Unknown') || 
    ' (Gender: ' || COALESCE(Gender, 'Not Specified') || 
    ') with customer id ' || COALESCE(Cust_Id, 'Unknown') || 
    ' from ' || COALESCE(State, 'Unknown Location')  ||  
    ' purchased a ' || COALESCE(Product, 'Item') || 
    ' of type ' || COALESCE(Prod_Type, 'General') || 
    ' on ' || COALESCE(Date_Of_Purchase, 'N/A') || 
    ' for an amount of ' || COALESCE(Amout, 0) AS text_to_embed
    -- Keep original columns for metadata filtering
FROM sales_data;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Optional: Clean text (lowercase, remove extra spaces)
    df['text_to_embed'] = df['text_to_embed'].str.strip().str.lower()
    
    return df

# Usage
prepared_df = extract_and_prepare_data('mykart.db')
#prepared_df.drop(columns=prepared_df.columns[0], axis=1, inplace=True)
prepared_df.set_index('Cust_Id', inplace=True)
print(prepared_df['text_to_embed'].head())
print("="*30)
print(prepared_df)