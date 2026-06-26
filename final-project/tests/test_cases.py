from typing import List, Dict


TEST_CASES: List[Dict] = [
    {
        "id": 1,
        "category": "基础概念查询",
        "question": "什么是ReLU激活函数？",
        "expected_intent": "knowledge_query",
        "expected_keywords": ["ReLU", "激活函数", "修正线性单元"],
        "description": "测试基础概念查询能力，应返回ReLU的定义和特点",
    },
    {
        "id": 2,
        "category": "原理理解",
        "question": "为什么Transformer能解决长序列依赖问题？",
        "expected_intent": "principle_derivation",
        "expected_keywords": ["Transformer", "自注意力", "长序列"],
        "description": "测试原理推导能力，应解释自注意力机制如何解决长距依赖",
    },
    {
        "id": 3,
        "category": "对比分析",
        "question": "CNN和RNN的区别和适用场景是什么？",
        "expected_intent": "comparison",
        "expected_keywords": ["CNN", "RNN", "卷积", "循环"],
        "description": "测试对比分析能力，应对比两种架构的核心特点和适用场景",
    },
    {
        "id": 4,
        "category": "复杂推理",
        "question": "深度学习中为什么会出现梯度消失问题？有哪些解决方法？",
        "expected_intent": "principle_derivation",
        "expected_keywords": ["梯度消失", "ReLU", "残差连接", "LSTM"],
        "description": "测试复杂推理能力，应解释原因并列出解决方法",
    },
    {
        "id": 5,
        "category": "无关问题拒绝",
        "question": "今天天气怎么样？",
        "expected_intent": "off_topic",
        "expected_keywords": [],
        "description": "测试安全护栏，应拒绝回答并提示仅支持深度学习问题",
    },
    {
        "id": 6,
        "category": "概念解释",
        "question": "什么是Transformer架构？它的核心组件是什么？",
        "expected_intent": "concept_explanation",
        "expected_keywords": ["Transformer", "自注意力", "多头注意力", "位置编码"],
        "description": "测试概念解释能力，应详细解释Transformer及其核心组件",
    },
    {
        "id": 7,
        "category": "多轮对话上下文",
        "question": "它和RNN有什么区别？",
        "expected_intent": "comparison",
        "expected_keywords": ["RNN"],
        "description": "测试多轮对话的上下文理解能力（需要在有前置对话时测试）",
    },
]
