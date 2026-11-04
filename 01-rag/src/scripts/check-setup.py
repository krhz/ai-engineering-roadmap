import os
from dotenv import load_dotenv
import psycopg
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv()

# 1. Base de datos + extensión vector
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    print("Postgres OK, pgvector", version[0])

# 2. Chat: Gemini con el SDK de OpenAI
# llm = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
llm = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["GROK_API_KEY"],
    max_retries=5,   # por defecto son 2, con espera exponencial
    timeout=60,
)
r = llm.chat.completions.create(
    model=os.environ["CHAT_MODEL"],
    messages=[{"role": "user", "content": "Responde solo con la palabra: listo"}],
)
print("Chat OK:", r.choices[0].message.content)

# 3. Embeddings
g = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # la llave de AI Studio
e = g.models.embed_content(
    model="gemini-embedding-001",
    contents="hola mundo",
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768),
)
print("Embeddings OK, dimensión:", len(e.embeddings[0].values))