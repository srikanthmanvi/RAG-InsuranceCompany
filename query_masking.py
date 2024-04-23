# pip install spacy
# python3 - m spacy download en_core_web_sm
from typing import List, Optional

import spacy
import re

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Compile regex patterns for performance
phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
date_pattern = re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b')
address_pattern = re.compile(r'\d+\s+[\w\s]+\,\s+[A-Za-z]+\,\s+[A-Z]{2}\s+\d{5}(-\d{4})?')
policy_number_pattern = re.compile(r'\bPLC\w*')

def mask_pii(text):
    """
    Masks Personally Identifiable Information (PII) in the given text using pre-defined regex patterns and spaCy's named entity recognition.

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
    text = policy_number_pattern.sub('[POLICY MASKED]', text)

    return text

class CustomPostProcessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        # Masks PII
        for n in nodes:
            n.node.set_content(mask_pii(n.text))
        return nodes



# Example text
text = "Hi my name is John Doe and my address is: 3424 Eden Ct, New York, NY 26543 and my email is johndoe@email.com. His birthday is 01/01/1990 and his phone is 123-456-7890."
conv="""
Customer: Hello, I'm William Anderson. My Date of Birth is 08/09/1933, Address is 606 Elm St, Dallas, TX 75201, and my Policy Number is PLC23653.
Agent: Good morning, William. How may I assist you today?
Customer: Hi, Jack. I have a question about my policy.
Customer: My basement flooded during heavy rainfall. Is water damage covered?
Agent: Let me review your policy for coverage related to water damage.
Agent: Yes, water damage from flooding is covered under your policy.
Customer: That's a relief. I'll need to schedule repairs as soon as possible.
Agent: We'll assist you with the claim process, William. Is there anything else I can help you with?
Customer: No, that's all for now. Thank you for your assistance, Jack.
Agent: You're welcome, William. Please feel free to reach out if you have any further questions or concerns.
Customer: I will. Have a great day!
Agent: You too, William. Take care.
"""

masked_text = maskPII(conv)
print(masked_text)
