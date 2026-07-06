import os
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

# Настройки подключения
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Lerkaa626",
    "host": "localhost",
    "port": 5433
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    sql_path = os.path.join("..", "data", "create_db.sql")
    if os.path.exists(sql_path):
        print(f"Выполняю скрипт создания таблиц из {sql_path}...")
        with open(sql_path, "r", encoding="utf-8") as f:
            cur.execute(f.read())
        conn.commit()
    else:
        print(f"Файл {sql_path} не найден")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    csv_path = "C:/Users/Lerik/OneDrive/Desktop/Uncode/search_product/backend/data/grocery_chain_data.csv" 

    df = pd.read_csv(csv_path)
    print(format(f"Начинаю загрузку {len(df)} строк в базу данных Hydra..."))

    for index, row in df.iterrows():
        text_representation = f"Product: {row['product_name']}, Aisle: {row['aisle']}, Store: {row['store_name']}"
        vector = model.encode(text_representation).tolist()
        
        sql_product = """
            INSERT INTO client_products (
                customer_id, store_name, transaction_date, aisle, product_name, 
                quantity, unit_price, total_amount, discount_amount, final_amount, loyalty_points,
                fts_vector
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
            RETURNING id;
        """
        
        cur.execute(sql_product, (
            int(row['customer_id']), row['store_name'], row['transaction_date'], row['aisle'], row['product_name'],
            int(row['quantity']), float(row['unit_price']), float(row['total_amount']), 
            float(row['discount_amount']), float(row['final_amount']), int(row['loyalty_points']),
            text_representation
        ))

        product_id = cur.fetchone()[0]
        
        sql_vector = """
            INSERT INTO product_embeddings (product_id, embedding)
            VALUES (%s, %s::vector);
        """
        cur.execute(sql_vector, (product_id, vector))

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()