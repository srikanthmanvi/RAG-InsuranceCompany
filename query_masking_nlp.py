# pip install spacy
# python3 - m spacy download en_core_web_sm
import re
from typing import List, Optional

import spacy
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from index import es_vector_store

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Compile regex patterns for performance
phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
date_pattern = re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b')
dob_pattern = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{1,2})(st|nd|rd|th),\s(\d{4})")
address_pattern = re.compile(r'\d+\s+[\w\s]+\,\s+[A-Za-z]+\,\s+[A-Z]{2}\s+\d{5}(-\d{4})?')
policy_number_pattern = re.compile(r"[A-Z]{3}\d{4}\.$")  # 3 characters followed by 4 digits, in our case

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

match = re.match(policy_number_pattern, "XYZ9876")
print(match)


def mask_pii(text):
    """
    Masks Personally Identifiable Information (PII) in the given
    text using pre-defined regex patterns and spaCy's named entity recognition.

    Args:
        text (str): The input text containing potential PII.

    Returns:
        str: The text with PII masked.
    """
    # Process the text with spaCy for NER
    doc = nlp(text)

    # Mask entities identified by spaCy NER (e.g First/Last Names etc)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            text = text.replace(ent.text, '[MASKED]')

    # Apply regex patterns after NER to avoid overlapping issues
    text = phone_pattern.sub('[PHONE MASKED]', text)
    text = email_pattern.sub('[EMAIL MASKED]', text)
    text = date_pattern.sub('[DATE MASKED]', text)
    text = address_pattern.sub('[ADDRESS MASKED]', text)
    text = dob_pattern.sub('[DOB MASKED]', text)
    text = policy_number_pattern.sub('[POLICY MASKED]', text)

    return text


class CustomPostProcessor(BaseNodePostprocessor):
    """
    Custom Postprocessor which masks Personally Identifiable Information (PII).
    PostProcessor is called on the Documents before they are sent to the LLM.
    """

    def _postprocess_nodes(
            self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # Masks PII
        for n in nodes:
            masked_text = mask_pii(n.text)
            print(masked_text)
            # n.node.set_content(mask_pii(n.text))
            n.node.set_content(masked_text)
        return nodes


# Use Public LLM to send user query and Related Documents
llm = OpenAI()

index = VectorStoreIndex.from_vector_store(es_vector_store)

# This query_engine, for a given user query retrieves top 10 similar documents from
# Elasticsearch vector database and sends the documents along with the user query to the LLM.
# Note that documents are sent as-is. So any PII/Sensitive data is sent to the LLM.
query_engine = index.as_query_engine(llm, similarity_top_k=10, node_postprocessors=[CustomPostProcessor()])

query = "Give me summary of water related claims that customers raised."
bundle = QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
result = query_engine.query(bundle)
print(result)
