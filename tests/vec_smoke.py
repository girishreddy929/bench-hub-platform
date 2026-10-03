import sqlite3, sqlite_vec, ollama

MODEL = "qwen3-embedding:0.6b"  # use the exact name shown by `ollama list`

c = sqlite3.connect(":memory:")
c.enable_load_extension(True)
sqlite_vec.load(c)
c.enable_load_extension(False)
print("vec version:", c.execute("select vec_version()").fetchone()[0])

texts = ["Azure ML", "Azure Machine Learning", "MLflow model tracking", "cooking pasta"]
emb = ollama.embed(model=MODEL, input=texts)["embeddings"]
print("dimension:", len(emb[0]))

blob = sqlite_vec.serialize_float32
for i in range(1, len(texts)):
    dist = c.execute("select vec_distance_cosine(?, ?)", (blob(emb[0]), blob(emb[i]))).fetchone()[0]
    print(f"Azure ML vs {texts[i]!r}: cosine similarity = {1 - dist:.3f}")