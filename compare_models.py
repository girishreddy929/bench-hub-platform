from src.ai import embeddings as e

PAIRS = [
    ("natural language processing", "NLP"),
    ("azure ml", "Azure Machine Learning"),
    ("azure ml", "MLflow"),
    ("azure ml", "cooking pasta"),
    ("sql", "PostgreSQL"),
    ("sql", "React"),
    ("feature store", "Iguazio MLRun"),
]
for model in ["qwen3-embedding:0.6b", "nomic-embed-text"]:
    print(model)
    for query, doc in PAIRS:
        q = e.embed_query(query, model=model)
        d = e.embed_documents([doc], model=model)[0]
        print(f"  {query!r} vs {doc!r}: {sum(a * b for a, b in zip(q, d)):.3f}")