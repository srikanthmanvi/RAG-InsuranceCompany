from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.core.postprocessor import NERPIINodePostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from index import es_vector_store

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Use Public LLM to send user query and Related Documents
llm = OpenAI()

ner_processor = NERPIINodePostprocessor()
index = VectorStoreIndex.from_vector_store(es_vector_store)

# This query_engine, for a given user query retrieves top 10 similar documents from
# Elasticsearch vector database and sends the documents along with the user query to the LLM.
# Note that documents masked using the NERPIINodePostprocessor so that PII/Sensitive data is not sent to the LLM.
query_engine = index.as_query_engine(llm, similarity_top_k=10, node_postprocessors=[ner_processor])

query = "Give me summary of fire related claims that customers raised."
bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)
