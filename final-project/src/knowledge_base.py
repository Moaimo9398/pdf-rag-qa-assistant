import os
import logging
from typing import List

from langchain_core.documents import Document

from src.config import PDF_PATH, CHROMA_PERSIST_DIR
from src.tools.pdf_parser import extract_text_from_pdf, extract_chapters
from src.tools.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, pdf_path: str = PDF_PATH):
        self.pdf_path = pdf_path
        self.retriever = VectorRetriever()
        self._built = False

    def build(self, force_rebuild: bool = False) -> bool:
        if not force_rebuild:
            if self.retriever.load_existing():
                logger.info("已加载现有向量知识库")
                self._built = True
                return True

        if not os.path.exists(self.pdf_path):
            logger.error(f"PDF文件不存在: {self.pdf_path}")
            return False

        try:
            logger.info(f"正在解析PDF: {self.pdf_path}")
            pages = extract_text_from_pdf(self.pdf_path)
            if not pages:
                logger.error("PDF解析结果为空")
                return False

            logger.info(f"成功解析 {len(pages)} 页")
            chapters = extract_chapters(pages)
            logger.info(f"识别到 {len(chapters)} 个章节")

            documents = self._chapters_to_documents(chapters)
            logger.info(f"文档块数: {len(documents)}")

            logger.info("正在构建向量知识库...")
            self.retriever.build_from_documents(documents)
            self._built = True
            logger.info("向量知识库构建完成")
            return True
        except Exception as e:
            logger.error(f"构建知识库失败: {e}")
            return False

    def _chapters_to_documents(self, chapters: List[dict]) -> List[Document]:
        documents = []
        for ch in chapters:
            text = ch["text"]
            if not text.strip():
                continue
            if len(text) > 450:
                chunks = self._split_long_text(text, max_len=400)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "page_num": ch["page_num"],
                            "chapter": ch["chapter"],
                            "source": os.path.basename(self.pdf_path),
                        },
                    ))
            else:
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "page_num": ch["page_num"],
                        "chapter": ch["chapter"],
                        "source": os.path.basename(self.pdf_path),
                    },
                ))
        return documents

    def _split_long_text(self, text: str, max_len: int = 400) -> List[str]:
        paragraphs = text.split("\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 > max_len and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + "\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text[:max_len]]

    def query(self, question: str, top_k: int = 5) -> List[Document]:
        if not self._built:
            logger.error("知识库尚未构建")
            return []
        return self.retriever.retrieve(question, top_k=top_k)

    def query_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Document]:
        if not self._built:
            logger.error("知识库尚未构建")
            return []
        return self.retriever.retrieve_by_keywords(keywords, top_k=top_k)

    def is_built(self) -> bool:
        return self._built

    def rebuild(self) -> bool:
        import shutil
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
        self._built = False
        return self.build(force_rebuild=True)
