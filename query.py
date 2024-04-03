from llama_index.core import VectorStoreIndex, QueryBundle, Response, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from index import es_vector_store
# Create index and query

local_llm = Ollama(model="mistral")
Settings.embed_model= OllamaEmbedding("mistral")

index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(local_llm, similarity_top_k=10)

query="Give me summary of water related issues"
bundle = QueryBundle(query,
                    embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)