# services/deepseek_service.py
import logging
import json
import os
import torch
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class DeepseekService:
    """Deepseek大模型服务，提供AI生成和理解功能 (本地模型版本)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Deepseek服务

        Args:
            config: 可选配置参数
        """
        self.config = config or {}

        # 从环境变量或配置中获取模型路径和设备信息
        self.model_path = os.getenv("DEEPSEEK_MODEL_PATH") or self.config.get("model_path")
        self.device_map = os.getenv("DEEPSEEK_DEVICE_MAP") or self.config.get("device_map", "auto")
        self.model_type = os.getenv("DEEPSEEK_MODEL_TYPE") or self.config.get("model_type", "deepseek-llm")
        self.load_8bit = os.getenv("DEEPSEEK_LOAD_8BIT", "False").lower() == "true" or self.config.get("load_8bit",
                                                                                                       False)
        self.load_4bit = os.getenv("DEEPSEEK_LOAD_4BIT", "False").lower() == "true" or self.config.get("load_4bit",
                                                                                                       False)

        # 模型和tokenizer
        self.model = None
        self.tokenizer = None

        # 获取可用GPU数量
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"检测到 {self.gpu_count} 个GPU设备")

        if not self.model_path:
            logger.warning("未设置Deepseek模型路径，请设置DEEPSEEK_MODEL_PATH环境变量或在配置中指定model_path")
        else:
            self._load_model()

    def _load_model(self):
        """加载Deepseek模型和tokenizer"""
        try:
            logger.info(f"正在加载Deepseek模型: {self.model_path}")
            logger.info(f"设备映射: {self.device_map}, GPU数量: {self.gpu_count}")
            logger.info(f"量化设置: load_8bit={self.load_8bit}, load_4bit={self.load_4bit}")

            if self.model_type == "deepseek-llm" or self.model_type == "deepseek-coder":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import gc

                # 清理GPU内存
                gc.collect()
                torch.cuda.empty_cache()

                # 加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

                # 配置模型参数
                # 设置为 bfloat16 或 float16 以减少显存使用，同时保持精度
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
                    0] >= 8 else torch.float16

                # 构建基本加载参数
                load_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch_dtype,
                }

                # 对于多GPU设置，使用自动或指定的设备映射
                if self.gpu_count > 1:
                    if self.device_map == "auto":
                        logger.info(f"使用自动设备映射分布模型到 {self.gpu_count} 个GPU")
                        load_kwargs["device_map"] = "auto"
                    else:
                        # 使用用户提供的自定义设备映射
                        logger.info(f"使用自定义设备映射: {self.device_map}")
                        load_kwargs["device_map"] = self.device_map
                else:
                    # 单GPU模式
                    logger.info("仅检测到1个GPU，使用单设备模式")
                    load_kwargs["device_map"] = 0

                # 应用量化配置 (如果启用)
                if self.load_8bit:
                    logger.info("使用8-bit量化加载模型")
                    load_kwargs["load_in_8bit"] = True
                elif self.load_4bit:
                    logger.info("使用4-bit量化加载模型")
                    load_kwargs["load_in_4bit"] = True
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[
                                                                     0] >= 8 else torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )

                # 加载模型
                logger.info(f"开始加载模型，参数: {load_kwargs}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )

                # 报告模型加载信息
                model_size = sum(p.numel() for p in self.model.parameters()) / 1e9
                gpu_memory = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in
                              range(self.gpu_count)] if self.gpu_count > 0 else []

                logger.info(f"Deepseek模型加载成功: {self.model_path}")
                logger.info(f"模型参数量: {model_size:.2f}B")
                if gpu_memory:
                    logger.info(f"GPU显存: {gpu_memory} GB")

                # 模型设备信息
                if hasattr(self.model, 'hf_device_map'):
                    logger.info(f"模型设备映射: {self.model.hf_device_map}")
            else:
                logger.error(f"不支持的模型类型: {self.model_type}")
                raise ValueError(f"不支持的模型类型: {self.model_type}")

        except Exception as e:
            logger.error(f"加载Deepseek模型时出错: {str(e)}")
            raise

    async def query(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, stream: bool = False) -> Union[
        Dict[str, Any], AsyncGenerator]:
        """
        向Deepseek模型发送查询

        Args:
            prompt: 提示文本
            temperature: 采样温度
            max_tokens: 最大生成的令牌数
            stream: 是否使用流式生成

        Returns:
            模型的响应，或流式生成的情况下返回生成器
        """
        if self.model is None or self.tokenizer is None:
            return {
                "error": "模型未加载",
                "content": None,
                "status": "error"
            }

        try:
            # 根据模型类型设置适当的格式
            if self.model_type == "deepseek-llm":
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 对于Deepseek Coder或其他模型，可能需要不同的格式
                formatted_prompt = prompt

            # 编码输入
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

            # 移动输入到合适的设备
            if torch.cuda.is_available():
                if hasattr(self.model, 'hf_device_map'):
                    # 模型分布在多个设备上，将输入移动到第一个设备
                    first_device = next(iter(self.model.hf_device_map.values()))
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}
                else:
                    # 单个设备
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # 配置生成参数
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id
            }

            # 对于较长输入，自动减少最大生成长度以适应显存
            if inputs["input_ids"].shape[1] > 3000:
                generation_config["max_new_tokens"] = min(max_tokens, 512)
                logger.info(f"输入较长，已减少最大生成长度至 {generation_config['max_new_tokens']}")

            # 使用流式生成
            if stream:
                # 返回流式生成器
                return self._stream_generate(inputs, generation_config)

            # 非流式生成 (一次性生成全部内容)
            return await self._batch_generate(inputs, generation_config)

        except Exception as e:
            logger.error(f"调用Deepseek模型时出错: {str(e)}")
            return {
                "error": f"调用模型时出错: {str(e)}",
                "content": None,
                "status": "error"
            }

    async def _batch_generate(self, inputs: Dict[str, Any], generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """非流式生成 - 一次性返回全部内容"""

        # 生成回答
        with torch.no_grad():
            # 检查并尝试优化显存
            try:
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # 清理显存并尝试重新生成
                    logger.warning("显存不足，尝试优化...")
                    torch.cuda.empty_cache()

                    # 如果仍有OOM错误，减少生成长度
                    generation_config["max_new_tokens"] = min(generation_config["max_new_tokens"], 256)
                    logger.info(f"减少最大生成长度至 {generation_config['max_new_tokens']}")

                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                else:
                    raise

        # 解码输出
        response_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return {
            "content": response_text.strip(),
            "status": "success"
        }

    async def _stream_generate(self, inputs: Dict[str, Any], generation_config: Dict[str, Any]):
        """流式生成 - 逐token返回内容"""

        # 确保我们使用TextIteratorStreamer
        from transformers import TextIteratorStreamer

        # 输入ID的长度 (用于提取新生成的token)
        input_length = inputs["input_ids"].shape[1]

        try:
            # 初始化streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, timeout=120)

            # 准备生成参数
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": generation_config.get("max_new_tokens", 1000),
                "temperature": generation_config.get("temperature", 0.7),
                "do_sample": generation_config.get("do_sample", True),
                "pad_token_id": generation_config.get("pad_token_id", self.tokenizer.eos_token_id)
            }

            # 在后台线程中运行生成
            import threading
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # 累积的文本
            accumulated_text = ""

            # 从streamer中迭代获取生成的token
            for new_text in streamer:
                accumulated_text += new_text
                yield {
                    "token": new_text,
                    "text": accumulated_text,
                    "status": "generating"
                }

            # 生成完成后返回完整结果
            yield {
                "token": "",
                "text": accumulated_text,
                "status": "complete"
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # 显存不足，尝试非流式生成
                logger.warning("流式生成时显存不足，回退到批量生成模式")
                torch.cuda.empty_cache()

                # 减少token数量并重试
                generation_config["max_new_tokens"] = min(generation_config["max_new_tokens"], 256)
                batch_result = await self._batch_generate(inputs, generation_config)

                yield {
                    "token": batch_result.get("content", ""),
                    "text": batch_result.get("content", ""),
                    "status": "complete",
                    "fallback": True
                }
            else:
                yield {
                    "error": str(e),
                    "status": "error"
                }
        except Exception as e:
            yield {
                "error": str(e),
                "status": "error"
            }

    async def summarize_document(self, text: str, max_length: int = 500, stream: bool = False) -> Dict[str, Any]:
        """
        使用Deepseek模型总结文档内容

        Args:
            text: 文档文本
            max_length: 摘要最大长度
            stream: 是否使用流式响应

        Returns:
            包含摘要的结果，或流式生成情况下返回生成器
        """
        # 截断过长的文档
        if len(text) > 12000:
            text = text[:12000] + "...(内容已截断)"

        prompt = f"""请对以下文档内容进行总结，生成一个简洁明了的摘要，不超过{max_length}字:

文档内容:
{text}

摘要:"""

        result = await self.query(prompt, temperature=0.3, max_tokens=max_length, stream=stream)

        if not stream and result.get("status") == "success":
            result["summary"] = result.get("content")

        return result

    async def extract_key_points(self, text: str, num_points: int = 5, stream: bool = False) -> Dict[str, Any]:
        """
        提取文档的关键点

        Args:
            text: 文档文本
            num_points: 提取的关键点数量
            stream: 是否使用流式响应

        Returns:
            包含关键点的结果，或流式生成情况下返回生成器
        """
        # 截断过长的文档
        if len(text) > 12000:
            text = text[:12000] + "...(内容已截断)"

        prompt = f"""请从以下文档中提取{num_points}个关键点，以简洁的项目符号形式呈现:

文档内容:
{text}

关键点:"""

        result = await self.query(prompt, temperature=0.3, stream=stream)

        if not stream and result.get("status") == "success":
            # 尝试解析关键点列表
            key_points_text = result.get("content", "")
            key_points = []

            # 简单解析带有符号的列表
            for line in key_points_text.split("\n"):
                line = line.strip()
                if line.startswith(("- ", "• ", "* ", "· ")):
                    key_points.append(line[2:])
                elif line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ", "0. ")):
                    key_points.append(line[3:])

            result["key_points"] = key_points

        return result

    async def answer_question(self, question: str, context: Optional[str] = None, stream: bool = False) -> Dict[
        str, Any]:
        """
        使用Deepseek回答问题

        Args:
            question: 问题文本
            context: 可选的上下文内容
            stream: 是否使用流式响应

        Returns:
            包含回答的结果，或流式生成情况下返回生成器
        """
        if context:
            # 截断过长的上下文
            if len(context) > 12000:
                context = context[:12000] + "...(内容已截断)"

            prompt = f"""根据以下内容回答问题:

背景内容:
{context}

问题: {question}

请提供详细、准确、全面的回答。"""
        else:
            prompt = f"""问题: {question}

请提供详细、准确、全面的回答。"""

        result = await self.query(prompt, temperature=0.7, stream=stream)

        if not stream and result.get("status") == "success":
            result["answer"] = result.get("content")

        return result

    async def generate_document_questions(self, text: str, num_questions: int = 5, stream: bool = False) -> Dict[
        str, Any]:
        """
        为文档生成问题，帮助用户理解文档内容

        Args:
            text: 文档文本
            num_questions: 生成的问题数量
            stream: 是否使用流式响应

        Returns:
            包含生成问题的结果，或流式生成情况下返回生成器
        """
        # 截断过长的文档
        if len(text) > 12000:
            text = text[:12000] + "...(内容已截断)"

        prompt = f"""根据以下文档内容，生成{num_questions}个有意义的问题，这些问题可以帮助读者理解文档的主要内容和关键观点:

文档内容:
{text}

请按照以下格式生成问题列表:
1. 问题1
2. 问题2
...
"""

        result = await self.query(prompt, temperature=0.7, stream=stream)

        if not stream and result.get("status") == "success":
            # 解析问题列表
            questions_text = result.get("content", "")
            questions = []

            # 简单解析带有编号的列表
            for line in questions_text.split("\n"):
                line = line.strip()
                # 匹配如 "1. ", "2. " 等开头的行
                if line and line[0].isdigit() and line[1:].startswith(". "):
                    questions.append(line[3:])

            result["questions"] = questions

        return result

    async def analyze_sentiment(self, text: str, stream: bool = False) -> Dict[str, Any]:
        """
        分析文本的情感倾向

        Args:
            text: 待分析的文本
            stream: 是否使用流式响应

        Returns:
            情感分析结果，或流式生成情况下返回生成器
        """
        # 截断过长的文本
        if len(text) > 5000:
            text = text[:5000] + "...(内容已截断)"

        prompt = """请分析以下文本的情感倾向，并将结果按照以下JSON格式返回:
{
  "sentiment": "positive|neutral|negative",
  "sentiment_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "explanation": "简要解释"
}

文本内容:
""" + text

        result = await self.query(prompt, temperature=0.2, stream=stream)

        if not stream and result.get("status") == "success":
            # 尝试从响应中提取JSON
            try:
                content = result.get("content", "")
                # 找到JSON部分
                json_start = content.find("{")
                json_end = content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    sentiment_data = json.loads(json_str)
                    result["sentiment_analysis"] = sentiment_data
                else:
                    result["sentiment_analysis"] = {"raw_output": content}
            except Exception as e:
                logger.error(f"解析情感分析结果时出错: {str(e)}")
                result["sentiment_analysis"] = {"error": str(e), "raw_output": result.get("content")}

        return result

    async def compare_documents(self, doc1: str, doc2: str, stream: bool = False) -> Dict[str, Any]:
        """
        比较两个文档的内容差异

        Args:
            doc1: 第一个文档的文本
            doc2: 第二个文档的文本
            stream: 是否使用流式响应

        Returns:
            比较结果，或流式生成情况下返回生成器
        """
        # 截断过长的文档
        if len(doc1) > 6000:
            doc1 = doc1[:6000] + "...(内容已截断)"
        if len(doc2) > 6000:
            doc2 = doc2[:6000] + "...(内容已截断)"

        prompt = f"""请比较以下两个文档的内容，分析它们的主要相似点和差异点:

文档1:
{doc1}

文档2:
{doc2}

请提供详细的比较分析，包括:
1. 主要相似点
2. 主要差异点
3. 内容覆盖范围的差异
4. 观点或立场的差异（如有）
"""

        result = await self.query(prompt, temperature=0.5, max_tokens=2000, stream=stream)

        if not stream and result.get("status") == "success":
            result["comparison"] = result.get("content")

        return result