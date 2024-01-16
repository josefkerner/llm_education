from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

from maia_gpt.chains.llm_chain import create_llm_chain
from maia_gpt.chains.qa_chains import create_stuff_documents_chain
from maia_gpt.config import CONFIG
from maia_gpt.logger import get_logger
from maia_gpt.models.chat_models import create_chat_llm
from maia_gpt.modules.firewall.firewall import PromptFirewall
from maia_gpt.prompts.chat_prompt import create_chat_prompt_template
from maia_gpt.storage.vector_store import VectorStore
from maia_gpt.utils.evaluation_utils import is_response_context_based
from maia_gpt.utils.openai_utils import log_openai_usage
from maia_gpt.utils.translation_utils import (
    create_language_detector,
    translate_response,
)

logger = get_logger("maia-gpt")

# Create the chat LLM model
chat_llm = create_chat_llm(backend_service=CONFIG.BACKEND_SERVICE)

# Create chains for question answering, response evaluation, and translation
chat_prompt = create_chat_prompt_template(CONFIG.QA_SYSTEM_PROMPT_TEMPLATE, human_template="{question}")
combine_documents_chain = create_stuff_documents_chain(llm=chat_llm, prompt=chat_prompt)

evaluate_response_prompt = create_chat_prompt_template(
    CONFIG.EVALUATE_RESPONSE_PROMPT_TEMPLATE, human_template="{response}"
)
evaluate_response_chain = create_llm_chain(llm=chat_llm, prompt=evaluate_response_prompt)

translation_prompt = create_chat_prompt_template(CONFIG.TRANSLATE_PROMPT_TEMPLATE, human_template="{text}")
translation_chain = create_llm_chain(llm=chat_llm, prompt=translation_prompt)

# Create a language detector to identify the language of questions and responses
language_detector = create_language_detector(CONFIG.LANGUAGE_DETECTOR, CONFIG.LINGUA_LANGUAGES)
prompt_firewall = PromptFirewall()


class Chatbot:
    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store
        self.qa = self.get_qa_chain(vector_store)

    def get_qa_chain(self, vector_store):
        retriever = vector_store.get_retriever(k=CONFIG.NUM_SECTIONS_TO_RETRIEVE, filter=None)
        qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            retriever=retriever,
            return_source_documents=True,
            input_key="question",
            output_key="answer",
        )
        return qa

    def get_chunks(self, question: str):
        qa_response = self.qa({"question": question})
        documents: List[Document] = qa_response["source_documents"]
        chunks = [{"section": doc.page_content, "metadata": str(doc.metadata)} for doc in documents]
        chunks_text = [chunk["section"] for chunk in chunks]
        return chunks_text