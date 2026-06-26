import os
import logging
import time
from typing import List, Optional

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import API_KEY, BASE_URL, EMBEDDING_MODEL, CHROMA_PERSIST_DIR, TOP_K

logger = logging.getLogger(__name__)

MAX_EMBEDDING_LENGTH = 450


def get_embedding_function():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )


class VectorRetriever:
    def __init__(self):
        self.embedding = get_embedding_function()
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._all_docs_cache = None

    def _get_or_create_client(self):
        if self.client is None:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    def _truncate_text(self, text: str, max_len: int = MAX_EMBEDDING_LENGTH) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len]

    def build_from_documents(self, documents: List[Document]):
        self._get_or_create_client()
        self._all_docs_cache = None
        try:
            self.client.delete_collection("dl_knowledge")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="dl_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

        texts = [self._truncate_text(doc.page_content) for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        batch_size = 20
        all_ids = []
        offset = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            for attempt in range(3):
                try:
                    embeddings = self.embedding.embed_documents(batch_texts)
                    ids = [f"doc_{offset + j}" for j in range(len(batch_texts))]
                    self.collection.add(
                        ids=ids,
                        documents=batch_texts,
                        embeddings=embeddings,
                        metadatas=batch_metas,
                    )
                    all_ids.extend(ids)
                    offset += len(batch_texts)
                    logger.info(f"已嵌入 {offset}/{len(texts)} 个文档块")
                    break
                except Exception as e:
                    logger.warning(f"嵌入批次 {i} 尝试 {attempt+1} 失败: {e}")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        logger.error(f"嵌入批次 {i} 彻底失败，跳过")
                        offset += len(batch_texts)

        logger.info(f"知识库构建完成，共 {len(all_ids)} 个文档块")
        return self.collection

    def load_existing(self) -> bool:
        self._get_or_create_client()
        self._all_docs_cache = None
        try:
            self.collection = self.client.get_collection("dl_knowledge")
            if self.collection.count() > 0:
                return True
        except Exception:
            pass
        return False

    def _embed_with_retry(self, text: str, max_retries: int = 2) -> list:
        for attempt in range(max_retries):
            try:
                return self.embedding.embed_query(text)
            except Exception as e:
                logger.warning(f"嵌入尝试 {attempt+1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return []

    def _get_all_docs(self) -> List[Document]:
        if self._all_docs_cache is not None:
            return self._all_docs_cache
        if self.collection is None:
            return []
        try:
            results = self.collection.get(include=["documents", "metadatas"])
            self._all_docs_cache = []
            for i, text in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                self._all_docs_cache.append(Document(page_content=text, metadata=dict(meta)))
            return self._all_docs_cache
        except Exception:
            return []

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        if self.collection is None:
            return []

        vector_results = self._vector_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)

        seen = set()
        merged = []
        for doc in vector_results + keyword_results:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        merged.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return merged[:top_k]

    def _vector_search(self, query: str, top_k: int) -> List[Document]:
        try:
            truncated_query = self._truncate_text(query, 200)
            query_embedding = self._embed_with_retry(truncated_query)
            if not query_embedding:
                return []
            count = self.collection.count()
            if count == 0:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            if results and results["documents"] and results["documents"][0]:
                for i, text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    metadata["relevance_score"] = round(1 - distance, 4)
                    docs.append(Document(page_content=text, metadata=metadata))
            return docs
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> List[Document]:
        all_docs = self._get_all_docs()
        if not all_docs:
            return []

        phrases = self._extract_phrases(query)
        if not phrases:
            return []

        scored = []
        for doc in all_docs:
            text_lower = doc.page_content.lower()
            hit_count = 0
            for phrase in phrases:
                if phrase.lower() in text_lower:
                    hit_count += len(phrase)
            if hit_count > 0:
                score = round(hit_count / sum(len(p) for p in phrases), 4)
                meta = dict(doc.metadata)
                meta["relevance_score"] = score
                meta["match_method"] = "keyword"
                scored.append(Document(page_content=doc.page_content, metadata=meta))

        scored.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return scored[:top_k]

    def _extract_phrases(self, text: str) -> List[str]:
        import re
        phrases = []
        en_words = re.findall(r'[a-zA-Z]{2,}', text)
        phrases.extend(en_words)
        zh_segments = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        for seg in zh_segments:
            if len(seg) >= 4:
                for length in [4, 3, 2]:
                    for i in range(len(seg) - length + 1):
                        phrases.append(seg[i:i+length])
            else:
                phrases.append(seg)
        return list(set(phrases))

    def retrieve_by_keywords(self, keywords: List[str], top_k: int = TOP_K) -> List[Document]:
        all_docs = self._get_all_docs()
        if not all_docs:
            return []

        scored = []
        for doc in all_docs:
            text_lower = doc.page_content.lower()
            hit_count = sum(1 for kw in keywords if kw.lower() in text_lower)
            if hit_count > 0:
                score = round(hit_count / max(len(keywords), 1), 4)
                meta = dict(doc.metadata)
                meta["relevance_score"] = score
                meta["match_method"] = "keyword"
                scored.append(Document(page_content=doc.page_content, metadata=meta))

        scored.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return scored[:top_k]
