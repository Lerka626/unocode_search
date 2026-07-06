CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE client_products (
    id SERIAL PRIMARY KEY,
    customer_id INT,
    store_name TEXT,
    transaction_date DATE,
    aisle TEXT,
    product_name TEXT,
    quantity INT,
    unit_price NUMERIC,
    total_amount NUMERIC,
    discount_amount NUMERIC,
    final_amount NUMERIC,
    loyalty_points INT,
    fts_vector tsvector
);

CREATE TABLE product_embeddings (
    product_id INT PRIMARY KEY,
    embedding vector(384)
) USING heap; 

CREATE INDEX fts_idx ON client_products USING gin(fts_vector);
CREATE INDEX hnsw_idx ON product_embeddings USING hnsw (embedding vector_cosine_ops);
