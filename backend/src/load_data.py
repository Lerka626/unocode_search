import os
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer

# Настройки подключения
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Lerkaa626"), 
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", 5432))
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # ИСПРАВЛЕНО: путь к .sql файлу относительно корня контейнера (/app)
    sql_path = os.path.join("data", "create_db.sql")
    if os.path.exists(sql_path):
        print(f"Выполняю скрипт создания таблиц из {sql_path}...")
        with open(sql_path, "r", encoding="utf-8") as f:
            cur.execute(f.read())
        conn.commit()
    else:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл {sql_path} не найден!")
        return  # Завершаем работу, если нет схемы БД

    print("Загружаю модель эмбеддингов all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ИСПРАВЛЕНО: убрали локальный путь C:/... и сделали его относительным для контейнера
    csv_path = os.path.join("data", "grocery_chain_data.csv") 

    if not os.path.exists(csv_path):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл с данными {csv_path} не найден!")
        return

    df = pd.read_csv(csv_path)
    print(f"Начинаю загрузку {len(df)} строк в базу данных...")

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
    print("=== [LOAD DATA SUCCESS] Все данные успешно загружены в базу! ===")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
