import json
import logging
import re
import time
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.config import API_KEY, BASE_URL, LLM_MODEL, TEMPERATURE, TOP_K
from src.knowledge_base import KnowledgeBase
from src.memory import ConversationMemory
from src.reflection import ReflectionModule
from src.tools.code_executor import CodeExecutor

logger = logging.getLogger(__name__)

OFF_TOPIC_RESPONSE = "我是深度学习知识库助手，只能回答与深度学习相关的问题。请您提问深度学习领域的知识，例如：什么是CNN？梯度消失如何解决？Transformer的原理是什么？"

DL_KEYWORDS = [
    "深度学习", "神经网络", "CNN", "RNN", "LSTM", "GRU", "Transformer",
    "BERT", "GPT", "激活函数", "ReLU", "Sigmoid", "Softmax", "Tanh",
    "梯度", "反向传播", "卷积", "池化", "全连接", "注意力", "自注意力",
    "优化器", "SGD", "Adam", "RMSprop", "过拟合", "正则化", "Dropout",
    "归一化", "残差", "ResNet", "RAG", "PyTorch", "TensorFlow",
    "损失函数", "学习率", "epoch", "batch", "训练", "推理",
    "编码器", "解码器", "位置编码", "预训练", "微调",
    "梯度消失", "梯度爆炸", "权重初始化", "数据增强",
    "Leaky", "收敛", "特征", "隐藏层", "输出层", "输入层",
]

INTENT_RULES = {
    "comparison": ["比较", "对比", "区别", "不同", "vs", "和.*区别", "与.*比较"],
    "principle_derivation": ["为什么", "原理", "如何解决", "怎么解决", "为什么能", "如何理解"],
    "code_example": ["代码", "实现", "编程", "写一个", "用PyTorch", "用Python"],
}

ANSWER_GENERATION_PROMPT = """你是深度学习专业助教。请根据以下从教材中检索到的内容，回答学生的问题。

要求：
1. 只根据检索到的内容回答，严禁编造文档中没有的知识点
2. 如果检索内容不足以完整回答问题，明确说明"根据现有教材内容，只能部分回答"
3. 回答要简洁专业，难点要完整解释
4. 在回答末尾标注来源章节

对话历史：
{history}

检索到的教材内容：
{context}

学生问题：{question}

请给出你的回答："""

REFLECTION_PROMPT = """审查以下回答质量。文档内容、问题和回答如下。

文档内容：{context}

问题：{question}

回答：{answer}

请判断：
A. 回答是否完全基于文档内容（未编造）？B. 是否完整回答了问题？C. 是否有幻觉？

只输出JSON，格式：{{"based_on_doc":true/false,"complete":true/false,"hallucination":true/false,"passed":true/false,"issue":""}}"""


class DeepLearnAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=TEMPERATURE,
        )
        self.kb = KnowledgeBase()
        self.memory = ConversationMemory()
        self.reflection = ReflectionModule()
        self.code_executor = CodeExecutor()

    def initialize(self, force_rebuild: bool = False) -> bool:
        return self.kb.build(force_rebuild=force_rebuild)

    def run(self, question: str) -> Dict:
        start_time = time.time()
        result = {
            "question": question,
            "intent": "",
            "answer": "",
            "sources": [],
            "related": [],
            "reflection": {},
            "error": None,
            "elapsed": 0.0,
        }

        try:
            intent = self._classify_intent_local(question)
            result["intent"] = intent
            logger.info(f"意图识别: {question} -> {intent}")

            if intent == "off_topic":
                result["answer"] = OFF_TOPIC_RESPONSE
                result["elapsed"] = round(time.time() - start_time, 2)
                return result

            docs = self._retrieve(question)
            if not docs:
                result["answer"] = "抱歉，在知识库中未找到与您问题相关的内容。请尝试换个问法，或确认您的问题与深度学习教材内容相关。"
                result["elapsed"] = round(time.time() - start_time, 2)
                return result

            result["sources"] = self._format_sources(docs)
            context = self._integrate_context(docs)

            answer = self._generate_answer(question, context)

            reflection_result = self.reflection.reflect(question, answer, context)
            result["reflection"] = reflection_result

            if not reflection_result.get("passed", True):
                logger.info("反思未通过，重新生成回答")
                answer = self._regenerate_answer(question, context, reflection_result.get("issue", ""))
                result["reflection"] = {"passed": True, "regenerated": True}

            if intent == "code_example":
                code_result = self._handle_code_request(question, answer)
                if code_result:
                    answer += f"\n\n--- 代码执行结果 ---\n{code_result}"

            result["answer"] = answer
            if reflection_result.get("hallucination"):
                result["answer"] += "\n\n⚠️ 注意：该回答可能包含未完全基于教材的内容，建议查阅原文确认。"
            result["answer"] += "\n\n📚 回答仅供学习参考，如有疑问请查阅权威教材。"

        except Exception as e:
            logger.error(f"智能体运行出错: {e}", exc_info=True)
            result["error"] = str(e)
            result["answer"] = f"抱歉，处理您的问题时出现了错误：{e}"

        result["elapsed"] = round(time.time() - start_time, 2)
        self.memory.add_turn(question, result["answer"])
        return result

    OFF_TOPIC_PATTERNS = [
        "天气", "气温", "温度", "吃饭", "做饭", "旅游", "电影", "音乐",
        "运动", "球赛", "股票", "房价", "八卦", "明星", "游戏", "小说",
        "购物", "化妆", "穿搭", "美食", "今天.+怎", "怎么.+样",
    ]

    def _classify_intent_local(self, question: str) -> str:
        q = question.lower()

        for pattern in self.OFF_TOPIC_PATTERNS:
            if re.search(pattern, q):
                return "off_topic"

        has_dl_keyword = any(kw.lower() in q for kw in DL_KEYWORDS)
        if not has_dl_keyword:
            return "off_topic"

        for intent, patterns in INTENT_RULES.items():
            for pattern in patterns:
                if re.search(pattern, q):
                    return intent

        return "knowledge_query"

    def _retrieve(self, question: str) -> List[Document]:
        docs = self.kb.query(question, top_k=TOP_K)
        context_keywords = self.memory.extract_context_keywords()
        if context_keywords:
            kw_docs = self.kb.query_by_keywords(context_keywords, top_k=3)
            seen = set()
            merged = []
            for doc in docs + kw_docs:
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
            return merged[:TOP_K]
        return docs

    def _integrate_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page_num", "?")
            chapter = doc.metadata.get("chapter", "未知章节")
            score = doc.metadata.get("relevance_score", "N/A")
            parts.append(f"[片段{i}] (第{page}页, {chapter}, 相关度:{score})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        history_text = self.memory.get_history_text(max_chars=800)
        prompt = ANSWER_GENERATION_PROMPT.format(
            history=history_text or "（无历史对话）",
            context=context,
            question=question,
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _regenerate_answer(self, question: str, context: str, issues: str) -> str:
        prompt = f"""之前的回答存在以下问题：{issues}

请你基于检索到的教材内容重新回答，注意纠正上述问题。

检索到的教材内容：
{context}

学生问题：{question}

请重新给出回答："""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _handle_code_request(self, question: str, answer: str) -> Optional[str]:
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
        if not code_blocks:
            return None
        result = self.code_executor.execute(code_blocks[0])
        if result["success"]:
            output = result["output"].strip()
            return f"执行成功：\n{output}" if output else "代码执行成功（无输出）"
        else:
            return f"执行失败：{result['error']}"

    def _format_sources(self, docs: List[Document]) -> List[Dict]:
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "page_num": doc.metadata.get("page_num", "?"),
                "chapter": doc.metadata.get("chapter", "未知"),
                "relevance_score": doc.metadata.get("relevance_score", "N/A"),
            })
        return sources

    def clear_memory(self):
        self.memory.clear()
