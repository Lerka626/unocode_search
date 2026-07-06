import time
import requests

BASE_URL = "http://127.0.0.1:8000/search"

TEST_QUERIES = [
    {"name": "Точное совпадение", "query": "Pasta"},
    {"name": "Поиск по смыслу/синонимам", "query": "something for Italian dinner"},
    {"name": "Категориальный запрос", "query": "Dairy products"}
]

def run_test_for_type(search_type: str, query: str):
    """Отправляет запрос на конкретный эндпоинт и замеряет время"""
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/{search_type}", params={"query": query})
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "Success",
                "time": elapsed_time,
                "results_count": len(data.get("search_results", [])),
                "sample_results": [r["product_name"] for r in data.get("search_results", [])[:2]],
                "llm_answer": data.get("llm_answer", "")[:150] + "..." # Обрезаем для компактности
            }
        else:
            return {"status": f"Error {response.status_code}", "time": elapsed_time, "results_count": 0, "sample_results": [], "llm_answer": ""}
    except Exception as e:
        return {"status": f"Connection Failed: {str(e)}", "time": 0, "results_count": 0, "sample_results": [], "llm_answer": ""}

def main():    
    for q in TEST_QUERIES:
        print(f"Сценарий: {q['name']} | Запрос: '{q['query']}'")
        
        # семантический поиск 
        semantic_res = run_test_for_type("semantic", q["query"])
        print(f"[Семантика] Статус: {semantic_res['status']} | Время: {semantic_res['time']:.4f} сек | Найдено товаров: {semantic_res['results_count']}")
        print(f"Примеры из БД: {semantic_res['sample_results']}")
        print(f"Ответ: {semantic_res['llm_answer']}\n")
        
        # векторный 
        vector_res = run_test_for_type("vector", q["query"])
        print(f"[Вектор] Статус: {vector_res['status']} | Время: {vector_res['time']:.4f} сек | Найдено товаров: {vector_res['results_count']}")
        print(f"Примеры из БД: {vector_res['sample_results']}")
        print(f"Ответ: {vector_res['llm_answer']}")

if __name__ == "__main__":
    main()
