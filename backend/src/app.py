import os
import requests
import subprocess
import asyncio
from fastapi import FastAPI, HTTPException
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Semantic vs Vector Search Comparison")

# Инициализация модели эмбеддингов для эндпоинтов поисков
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Lerkaa626"), 
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", 5432))
}

OPENROUTER_API_KEY = "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"

def run_migration_in_background():
    """
    Фоновая функция для запуска load_data.py.
    Убран таймаут, чтобы тяжелый процесс генерации эмбеддингов для строк CSV 
    мог завершиться без принудительной остановки.
    """
    print("=== [BACKGROUND] Начался фоновый запуск скрипта load_data.py ===")
    try:
        # Запускаем скрипт без timeout=60
        result = subprocess.run(["python", "src/load_data.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("=== [BACKGROUND SUCCESS] База данных успешно проверена/инициализирована ===")
            print(result.stdout)
        else:
            print("=== [BACKGROUND ERROR] Скрипт load_data.py завершился с ошибкой ===")
            print(result.stderr)
    except Exception as e:
        print(f"=== [BACKGROUND EXCEPTION] Не удалось запустить скрипт load_data.py: {e} ===")

# --- Автоматический запуск скрипта миграции при старте приложения ---
@app.on_event("startup")
def startup_init_db():
    print("=== [STARTUP] FastAPI успешно стартует. Запускаем миграцию в фоновом потоке... ===")
    
    # Запускаем выполнение функции в отдельном потоке (executor),
    # чтобы не блокировать цикл событий (event loop) самого FastAPI
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_migration_in_background)
# --------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def get_llm_answer(context: str, question: str) -> str:
    if not OPENROUTER_API_KEY:
        return "Ошибка: Не указан API-ключ OpenRouter в app.py"
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Ты — ИИ-ассистент магазина. Ответь на вопрос пользователя, используя только предоставленный контекст из базы данных.
Контекст:
{context}

Вопрос: {question}
Ответ:"""

    payload = {
        "model": "openrouter/free",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"OpenRouter вернул статус {response.status_code}: {response.text}")
            return f"Ошибка OpenRouter (Статус {response.status_code}): {response.text[:100]}"
        
        res_json = response.json()
        return res_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"Ошибка сети при вызове OpenRouter: {e}")
        return f"Ошибка сети: {str(e)[:50]}"

@app.get("/search/semantic")
def search_semantic(query: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        sql_query = """
            SELECT product_name, aisle, store_name, unit_price 
            FROM client_products 
            WHERE fts_vector @@ plainto_tsquery('english', %s)
            LIMIT 5;
        """
        cur.execute(sql_query, (query,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            return {"search_results": [], "llm_answer": "Ничего не найдено по текстовому поиску."}
        
        context = "\n".join([f"Товар: {r['product_name']}, Категория: {r['aisle']}, Цена: {r['unit_price']}" for r in rows])
        llm_answer = get_llm_answer(context, query)
        
        return {"search_results": rows, "llm_answer": llm_answer}
    except Exception as e:
        print(f"Ошибка в эндпоинте /search/semantic: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/vector")
def search_vector(query: str):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        sql_query = """
            SELECT p.product_name, p.aisle, p.store_name, p.unit_price 
            FROM product_embeddings v
            JOIN client_products p ON v.product_id = p.id
            ORDER BY v.embedding <=> %s::vector 
            LIMIT 5;
        """
        cur.execute(sql_query, (query_embedding,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            return {"search_results": [], "llm_answer": "Ничего не найдено по векторному поиску."}
        
        context = "\n".join([f"Товар: {r['product_name']}, Категория: {r['aisle']}, Цена: {r['unit_price']}" for r in rows])
        llm_answer = get_llm_answer(context, query)
        
        return {"search_results": rows, "llm_answer": llm_answer}
    except Exception as e:
        print(f"Ошибка в эндпоинте /search/vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))
