"""
rag_pipeline.py
Builds three FAISS vector stores (health / nutrition / training)
and exposes a unified retriever.

"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    EMBED_MODEL, RAG_TOP_K,
    HEALTH_CORPUS, NUTRITION_CORPUS, TRAINING_CORPUS,
    INDEX_DIR,
)
from data_ingestion import load_corpus

log = logging.getLogger(__name__)

DOMAIN_MAP = {
    "health":    HEALTH_CORPUS,
    "nutrition": NUTRITION_CORPUS,
    "training":  TRAINING_CORPUS,
}

INDEX_PATHS = {
    domain: os.path.join(INDEX_DIR, domain)
    for domain in DOMAIN_MAP
}


class RAGPipeline:

    def __init__(self, embed_model: str = EMBED_MODEL):
        log.info(f"Loading embedding model: {embed_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._stores: Dict[str, FAISS] = {}


    def build_indexes(self, force_rebuild: bool = False) -> None:
        for domain, corpus_path in DOMAIN_MAP.items():
            idx_path = INDEX_PATHS[domain]
            if not force_rebuild and os.path.exists(idx_path):
                log.info(f"Loading existing {domain} index from {idx_path}")
                self._stores[domain] = FAISS.load_local(
                    idx_path, self.embeddings, allow_dangerous_deserialization=True
                )
            else:
                if not os.path.exists(corpus_path):
                    raise FileNotFoundError(
                        f"Corpus not found: {corpus_path}\n"
                        "Run: python data_ingestion.py  to build corpora first."
                    )
                log.info(f"Building {domain} index from {corpus_path}…")
                docs = load_corpus(corpus_path)
                store = FAISS.from_documents(docs, self.embeddings)
                Path(idx_path).mkdir(parents=True, exist_ok=True)
                store.save_local(idx_path)
                self._stores[domain] = store
                log.info(f"  {domain} index built with {len(docs)} chunks.")

    def _get_store(self, domain: str) -> FAISS:
        if domain not in self._stores:
            raise RuntimeError(
                f"Domain '{domain}' not loaded. Call build_indexes() first."
            )
        return self._stores[domain]

    def retrieve(
        self,
        query: str,
        domain: str,
        k: int = RAG_TOP_K,
    ) -> List[Document]:
        store = self._get_store(domain)
        return store.similarity_search(query, k=k)

    def retrieve_with_scores(
        self,
        query: str,
        domain: str,
        k: int = RAG_TOP_K,
    ) -> List[tuple]:
        store = self._get_store(domain)
        return store.similarity_search_with_score(query, k=k)

    def retrieve_multi_domain(
        self,
        query: str,
        domains: List[str],
        k_per_domain: int = 2,
    ) -> List[Document]:
        results = []
        for domain in domains:
            try:
                docs = self.retrieve(query, domain, k=k_per_domain)
                results.extend(docs)
            except RuntimeError as e:
                log.warning(str(e))
        return results

    @staticmethod
    def show_evidence(
        query: str,
        docs: List[Document],
        max_chars: int = 300,
    ) -> None:
        print("RAG EVIDENCE")
        print(f"Query : {query}")
        print(f"Chunks: {len(docs)}\n")
        for i, d in enumerate(docs, 1):
            doc_id = d.metadata.get("doc_id", "N/A")
            domain = d.metadata.get("domain", "N/A")
            source = d.metadata.get("source", "N/A")
            title  = d.metadata.get("title", "")
            snippet = d.page_content[:max_chars].strip()
            if len(d.page_content) > max_chars:
                snippet += "…"
            print(f"[{i}] doc_id={doc_id} | domain={domain}")
            print(f"     title={title}")
            print(f"     source={source}")
            print(f"     snippet: {snippet}\n")

    @staticmethod
    def build_context_string(docs: List[Document]) -> str:
        blocks = []
        for d in docs:
            doc_id = d.metadata.get("doc_id", "unknown")
            blocks.append(f"[{doc_id}]\n{d.page_content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def extract_evidence_list(docs: List[Document]) -> List[Dict]:
        return [
            {
                "doc_id": d.metadata.get("doc_id", "N/A"),
                "title":  d.metadata.get("title", ""),
                "source": d.metadata.get("source", ""),
                "domain": d.metadata.get("domain", ""),
                "text":   d.page_content,
            }
            for d in docs
        ]
