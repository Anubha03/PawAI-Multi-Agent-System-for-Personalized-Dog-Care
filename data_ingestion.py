"""
data_ingestion.py

Fetches real data for all three RAG corpora:
  - Health Monitoring  → PubMed (canine health / veterinary research)
  - Nutrition          → PubMed + AKC/FDA/ASPCA public pages
  - Training/Behavior  → PubMed (Applied Animal Behaviour Science papers)

Output: JSONL files in ./corpus/ with fields  {doc_id, text, source, domain}
Each line = one retrievable chunk.
    
"""

import os
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

from config import (
    NCBI_EMAIL, NCBI_API_KEY, PUBMED_MAX_IDS,
    PUBMED_FETCH_BATCH_SIZE, PUBMED_SLEEP,         
    PUBMED_HEALTH_QUERIES, PUBMED_NUTRITION_QUERIES, PUBMED_TRAINING_QUERIES,
    NUTRITION_URLS, TRAINING_URLS, HEALTH_URLS,
    CORPUS_DIR, HEALTH_CORPUS, NUTRITION_CORPUS, TRAINING_CORPUS,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def _make_doc_id(text: str, prefix: str) -> str:
    return f"{prefix}_{hashlib.md5(text[:120].encode()).hexdigest()[:8]}"


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind('. ', start, end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 60]


def _save_jsonl(records: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(records)} chunks → {path}")

class PubMedFetcher:

    BASE_SEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    BASE_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(self, email: str = NCBI_EMAIL, api_key: str = NCBI_API_KEY):
        self.email    = email
        self.api_key  = api_key
        self.session  = requests.Session()
        self.session.headers.update({"User-Agent": f"DogOwnerAgent/1.0 ({email})"})
      
    def _params(self, extra: dict) -> dict:
        p = {"email": self.email, "retmode": "json"}
        if self.api_key:
            p["api_key"] = self.api_key
        p.update(extra)
        return p

    def search(self, query: str, max_results: int = PUBMED_MAX_IDS,
               extra_filter: str = "animals[MeSH]") -> List[str]:
        full_query = f"({query}) AND {extra_filter}" if extra_filter else query
        params = self._params({
            "db":     "pubmed",
            "term":   full_query,
            "retmax": max_results,
            "sort":   "relevance",
        })
        try:
            resp = self.session.get(self.BASE_SEARCH, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            log.warning(f"PubMed search failed for '{query}': {e}")
            return []

    def _fetch_batch(self, pmids: List[str]) -> List[Dict[str, str]]:
        params = {
            "db":      "pubmed",
            "id":      ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            resp = self.session.get(self.BASE_FETCH, params=params, timeout=30)
            resp.raise_for_status()
            return self._parse_xml_abstracts(resp.text)
        except Exception as e:
            log.warning(f"PubMed batch fetch failed ({len(pmids)} IDs): {e}")
            return []

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
      
        if not pmids:
            return []

        results: List[Dict[str, str]] = []
        for i in range(0, len(pmids), PUBMED_FETCH_BATCH_SIZE):
            batch = pmids[i : i + PUBMED_FETCH_BATCH_SIZE]
            log.debug(f"    efetch batch {i // PUBMED_FETCH_BATCH_SIZE + 1}: {len(batch)} IDs")
            results.extend(self._fetch_batch(batch))
            time.sleep(PUBMED_SLEEP)   

        return results

    @staticmethod
    def _parse_xml_abstracts(xml_text: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(xml_text, "lxml")
        articles = []
        for article in soup.find_all("pubmedarticle"):
            pmid_tag  = article.find("pmid")
            title_tag = article.find("articletitle")
            abstract  = article.find("abstracttext")
            if not (pmid_tag and title_tag):
                continue
            pmid    = pmid_tag.get_text(strip=True)
            title   = title_tag.get_text(strip=True)
            ab_text = abstract.get_text(" ", strip=True) if abstract else ""
            if ab_text:
                articles.append({
                    "pmid":   pmid,
                    "title":  title,
                    "text":   f"{title}. {ab_text}",
                    "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
        return articles

    def fetch_domain(self, queries: List[str], domain_prefix: str,
                     max_per_query: int = PUBMED_MAX_IDS) -> List[Dict]:
    
        seen_pmids: set = set()
        all_articles: List[Dict] = []

        for query in queries:
            log.info(f"  PubMed search: '{query}'")
            pmids     = self.search(query, max_results=max_per_query)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            if new_pmids:
                articles = self.fetch_abstracts(new_pmids)   
                all_articles.extend(articles)
                log.info(f"    → {len(articles)} abstracts fetched")
              
            time.sleep(PUBMED_SLEEP)

        records: List[Dict] = []
        for art in all_articles:
            chunks = _chunk_text(art["text"])
            for chunk in chunks:
                doc_id = _make_doc_id(chunk, domain_prefix)
                records.append({
                    "doc_id": doc_id,
                    "text":   chunk,
                    "source": art["source"],
                    "pmid":   art["pmid"],
                    "title":  art["title"],
                    "domain": domain_prefix,
                })

        log.info(f"  Total chunks for {domain_prefix}: {len(records)}")
        return records

class WebScraper:

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; DogOwnerAgentBot/1.0; "
            "+https://github.com/example/dog-owner-agent)"
        )
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def scrape(self, url: str, domain_prefix: str) -> List[Dict]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"Scrape failed {url}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")

        for tag in soup(["nav", "footer", "script", "style", "header",
                         "aside", "form", "button", "iframe"]):
            tag.decompose()

        main = (
            soup.find("article") or
            soup.find("main") or
            soup.find("div", class_=re.compile(r"(content|article|post|body)", re.I)) or
            soup.body
        )
        if not main:
            return []

        raw_text = main.get_text(separator=" ", strip=True)
        raw_text = re.sub(r'\s+', ' ', raw_text)

        if len(raw_text) < 200:
            return []

        chunks  = _chunk_text(raw_text)
        records = []
        for chunk in chunks:
            doc_id = _make_doc_id(chunk, domain_prefix)
            records.append({
                "doc_id": doc_id,
                "text":   chunk,
                "source": url,
                "title":  soup.title.string.strip() if soup.title else url,
                "domain": domain_prefix,
            })

        log.info(f"  {url} → {len(records)} chunks")
        return records

    def scrape_many(self, urls: List[str], domain_prefix: str) -> List[Dict]:
        all_records = []
        seen_ids: set = set()
        for url in urls:
            recs = self.scrape(url, domain_prefix)
            for r in recs:
                if r["doc_id"] not in seen_ids:
                    seen_ids.add(r["doc_id"])
                    all_records.append(r)
            time.sleep(1.0)   
        return all_records

def _build_structured_records(items: List[Dict], prefix: str) -> List[Dict]:
    records = []
    for item in items:
        chunks = _chunk_text(item["text"])
        for chunk in chunks:
            records.append({
                "doc_id": _make_doc_id(chunk, prefix),
                "text":   chunk,
                "source": "expert_structured_knowledge",
                "title":  item["topic"],
                "domain": prefix,
            })
    return records

def build_all_corpora(
    skip_pubmed: bool = False,
    skip_web: bool = False,
) -> None:
   
    Path(CORPUS_DIR).mkdir(parents=True, exist_ok=True)
    fetcher = PubMedFetcher()
    scraper = WebScraper()

    log.info("Building HEALTH corpus")
    health_records: List[Dict] = []
    
    if not skip_pubmed:
        log.info("  Fetching PubMed health articles")
        health_records += fetcher.fetch_domain(PUBMED_HEALTH_QUERIES, "health")

    if not skip_web:
        log.info("  Scraping health web pages…")
        health_records += scraper.scrape_many(HEALTH_URLS, "health")

    _deduplicate_and_save(health_records, HEALTH_CORPUS)

    log.info("Building NUTRITION corpus")
    nutrition_records: List[Dict] = []
    
    if not skip_pubmed:
        log.info("  Fetching PubMed nutrition articles…")
        nutrition_records += fetcher.fetch_domain(PUBMED_NUTRITION_QUERIES, "nutrition")

    if not skip_web:
        log.info("  Scraping nutrition web pages…")
        nutrition_records += scraper.scrape_many(NUTRITION_URLS, "nutrition")

    _deduplicate_and_save(nutrition_records, NUTRITION_CORPUS)

    log.info("=== Building TRAINING corpus ===")
    training_records: List[Dict] = []
  
    if not skip_pubmed:
        log.info("  Fetching PubMed behavior/training articles…")
        training_records += fetcher.fetch_domain(PUBMED_TRAINING_QUERIES, "training")

    if not skip_web:
        log.info("  Scraping training web pages…")
        training_records += scraper.scrape_many(TRAINING_URLS, "training")

    _deduplicate_and_save(training_records, TRAINING_CORPUS)

    log.info("✅  All corpora built successfully.")


def _deduplicate_and_save(records: List[Dict], path: str) -> None:
    seen   = set()
    unique = []
    for r in records:
        if r["doc_id"] not in seen:
            seen.add(r["doc_id"])
            unique.append(r)
    _save_jsonl(unique, path)
    log.info(f"  Final unique chunks: {len(unique)}")

def load_corpus(path: str):
    from langchain_core.documents import Document
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj  = json.loads(line)
            text = obj.pop("text")
            docs.append(Document(page_content=text, metadata=obj))
    return docs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Dog Owner Agent RAG corpora")
    parser.add_argument("--skip-pubmed", action="store_true", help="Skip PubMed API calls")
    parser.add_argument("--skip-web",    action="store_true", help="Skip web scraping")
    args = parser.parse_args()
    build_all_corpora(skip_pubmed=args.skip_pubmed, skip_web=args.skip_web)
