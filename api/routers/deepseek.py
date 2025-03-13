# api/routers/deepseek.py
import logging
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, Body, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from typing import Dict, List, Any, Optional, AsyncGenerator

from api.dependencies.services import get_deepseek_service, get_document_service, get_search_service
from api.schemas.common import ErrorResponse
from services.deepseek_service import DeepseekService
from services.document_service import DocumentService
from services.search_service import SearchService
from pydantic import BaseModel
import re
import os

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()


class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str
    context: Optional[str] = None
    stream: bool = False
    # 不添加额外字段，保持简单


class SummarizeRequest(BaseModel):
    """文档总结请求模型"""
    text: str
    max_length: Optional[int] = 500
    stream: bool = False


class KeyPointsRequest(BaseModel):
    """提取关键点请求模型"""
    text: str
    num_points: Optional[int] = 5
    stream: bool = False


class GenerateQuestionsRequest(BaseModel):
    """生成问题请求模型"""
    text: str
    num_questions: Optional[int] = 5
    stream: bool = False


class SentimentAnalysisRequest(BaseModel):
    """情感分析请求模型"""
    text: str
    stream: bool = False


class CompareDocumentsRequest(BaseModel):
    """文档比较请求模型"""
    doc1: str
    doc2: str
    stream: bool = False

# 添加新的模型类
class KnowledgeBaseQuestionRequest(BaseModel):
    """基于知识库的问题请求模型"""
    question: str
    top_k: int = 3  # 检索的文档数量
    stream: bool = False
    filter_threshold: float = 0.6  # 相关性过滤阈值


@router.post("/knowledge-base/answer")
async def answer_from_knowledge_base(
        request: KnowledgeBaseQuestionRequest,
        deepseek_service: DeepseekService = Depends(get_deepseek_service),
        search_service: SearchService = Depends(get_search_service)
):
    """基于知识库内容回答问题"""
    try:
        logger.info(f"收到知识库问答请求: {request.question}")

        # 步骤1: 使用搜索服务查询相关文档
        logger.info(f"开始搜索相关文档，top_k={request.top_k}, filter_threshold={request.filter_threshold}")
        search_results = search_service.search(
            query=request.question,
            top_k=request.top_k,
            search_type="hybrid"  # 使用混合搜索获取最佳结果
        )

        # 调试: 打印搜索结果
        logger.info(f"搜索结果: {search_results}")

        # 检查结果数量
        result_count = len(search_results.get("results", []))
        logger.info(f"搜索结果数量: {result_count}")

        # 步骤2: 过滤结果并组装上下文
        context_parts = []
        relevant_docs = []

        for result in search_results.get("results", []):
            # 记录搜索结果的分数
            score = result.get("score", 0)
            logger.info(f"检查搜索结果，分数={score}, 阈值={request.filter_threshold}")

            # 根据相关性分数过滤
            if score >= request.filter_threshold:
                # 添加文档内容到上下文
                text = result.get("text", "")
                logger.info(f"找到匹配结果，文本长度: {len(text)}")

                if text:
                    context_parts.append(text)

                    # 收集相关文档的元数据用于返回
                    metadata = result.get("metadata", {})
                    logger.info(f"添加相关文档，元数据: {metadata}")

                    relevant_docs.append({
                        "score": score,
                        "metadata": metadata,
                        "document_id": metadata.get("document_id", ""),
                        "title": metadata.get("title", "未知文档")
                    })
            else:
                logger.info(f"结果分数低于阈值，不添加到上下文")

        # 如果没有找到相关文档
        if not context_parts:
            logger.warning(f"未找到相关文档，query={request.question}")
            return {
                "answer": "我所学的知识中找不到与您问题相关的信息。请尝试重新表述您的问题，或者询问其他方面的问题。",
                "status": "success",
                "relevant_documents": [],
                "has_knowledge": False
            }

        # 调试: 打印上下文长度
        logger.info(f"找到 {len(context_parts)} 个相关段落，总字符数: {sum(len(part) for part in context_parts)}")

        # 步骤3: 组装上下文和提示
        combined_context = "\n\n---\n\n".join(context_parts)

        # 构建提示
        prompt = f"""请根据提供的知识库内容回答用户的问题。如果知识库内容不足以完整回答问题，请明确指出，并仅基于知识库中的信息提供部分回答。

知识库内容:
{combined_context}

用户问题: {request.question}

请提供详细、准确、全面的回答，并引用知识库中的信息支持您的回答。"""

        logger.info(f"构建提示完成，提示长度: {len(prompt)}")

        # 步骤4: 调用Deepseek服务
        # 对于流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.query(
                prompt=prompt,
                temperature=0.3,  # 使用较低的温度以保持忠实于知识库内容
                stream=True
            )

            # 添加文档信息
            docs_info_sent = False

            # 定义流式响应事件生成器
            async def event_generator():
                nonlocal docs_info_sent

                # 首先发送相关文档信息
                if relevant_docs and not docs_info_sent:
                    docs_info_event = {
                        "relevant_documents": relevant_docs,
                        "has_knowledge": True,
                        "status": "info"
                    }
                    yield f"data: {json.dumps(docs_info_event, ensure_ascii=False)}\n\n"
                    docs_info_sent = True

                # 然后发送生成的token
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        logger.info(f"调用Deepseek生成答案")
        result = await deepseek_service.query(
            prompt=prompt,
            temperature=0.3,
            stream=False
        )

        if result.get("status") == "error":
            logger.error(f"Deepseek服务返回错误: {result.get('error')}")
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        # 组装结果
        result.update({
            "relevant_documents": relevant_docs,
            "has_knowledge": True,
            "answer": result.get("content", "")  # 将内容映射为answer字段以保持一致性
        })

        logger.info(f"成功返回答案，长度: {len(result.get('answer', ''))}")
        return result

    except Exception as e:
        logger.error(f"基于知识库回答问题失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"基于知识库回答问题失败: {str(e)}")

# 为了兼容旧API，添加一个参数控制是否使用改进的流式返回方式
@router.post("/answer/v2")
async def answer_question_v2(
        request: QuestionRequest,
        mode: str = Query("chunk", description="流式响应模式: token(每个token单独发送) 或 chunk(合并多个token成块)"),
        buffer_size: int = Query(5, description="在chunk模式下累积的token数"),
        deepseek_service: DeepseekService = Depends(get_deepseek_service)
):
    """使用Deepseek回答问题，支持优化的流式响应"""
    if not request.stream:
        # 非流式请求，直接使用原来的处理方法
        return await answer_question(request, deepseek_service)

    try:
        # 获取流式生成器
        stream_generator = await deepseek_service.answer_question(
            question=request.question,
            context=request.context,
            stream=True
        )

        # 流式响应处理 - 支持两种模式
        last_sent_text = ""
        remove_special = request.remove_special_tokens
        token_buffer = []

        async def event_generator():
            nonlocal last_sent_text, token_buffer

            async for token_data in stream_generator:
                if token_data.get("status") == "error":
                    yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                    break

                # 获取完整的累积文本
                full_text = token_data.get("text", "")
                token = token_data.get("token", "")

                # 清理特殊标记 (如果需要)
                if remove_special:
                    full_text = clean_model_output(full_text)
                    # 对单个token的清理可能不可靠，所以我们计算差异
                    token = full_text[len(last_sent_text):]

                if mode == "token":
                    # Token模式: 每个token单独发送
                    if token:
                        response_data = {
                            "token": token,
                            "text": full_text,
                            "status": token_data.get("status", "generating")
                        }
                        yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                        last_sent_text = full_text

                else:  # chunk模式
                    # 累积token
                    if token:
                        token_buffer.append(token)

                    # 当积累了足够的token或者是最后一个token时发送
                    if len(token_buffer) >= buffer_size or token_data.get("status") == "complete":
                        if token_buffer:
                            chunk = "".join(token_buffer)
                            response_data = {
                                "chunk": chunk,
                                "text": full_text,
                                "status": token_data.get("status", "generating")
                            }
                            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                            last_sent_text = full_text
                            token_buffer = []

            # 最后发送完成标记
            yield "data: [DONE]\n\n"

        # 返回SSE流式响应
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"答疑失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"答疑失败: {str(e)}")


@router.post("/knowledge-base/answer")
async def answer_from_knowledge_base(
        request: KnowledgeBaseQuestionRequest,
        deepseek_service: DeepseekService = Depends(get_deepseek_service),
        search_service: SearchService = Depends(get_search_service)
):
    """基于知识库内容回答问题"""
    try:
        # 步骤1: 使用搜索服务查询相关文档
        search_results = search_service.search(
            query=request.question,
            top_k=request.top_k,
            search_type="hybrid"  # 使用混合搜索获取最佳结果
        )

        # 步骤2: 过滤结果并组装上下文
        context_parts = []
        relevant_docs = []

        for result in search_results.get("results", []):
            # 根据相关性分数过滤
            if result.get("score", 0) >= request.filter_threshold:
                # 添加文档内容到上下文
                text = result.get("text", "")
                if text:
                    context_parts.append(text)

                    # 收集相关文档的元数据用于返回
                    metadata = result.get("metadata", {})
                    relevant_docs.append({
                        "score": result.get("score", 0),
                        "metadata": metadata,
                        "document_id": metadata.get("document_id", ""),
                        "title": metadata.get("title", "未知文档")
                    })

        # 如果没有找到相关文档
        if not context_parts:
            return {
                "answer": "我在知识库中找不到与您问题相关的信息。请尝试重新表述您的问题，或者询问其他方面的问题。",
                "status": "success",
                "relevant_documents": [],
                "has_knowledge": False
            }

        # 步骤3: 组装上下文和提示
        combined_context = "\n\n---\n\n".join(context_parts)

        # 构建提示
        prompt = f"""请根据提供的知识库内容回答用户的问题。如果知识库内容不足以完整回答问题，请明确指出，并仅基于知识库中的信息提供部分回答。

知识库内容:
{combined_context}

用户问题: {request.question}

请提供详细、准确、全面的回答，并引用知识库中的信息支持您的回答。"""

        # 步骤4: 调用Deepseek服务
        # 对于流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.query(
                prompt=prompt,
                temperature=0.3,  # 使用较低的温度以保持忠实于知识库内容
                stream=True
            )

            # 添加文档信息
            docs_info_sent = False

            # 定义流式响应事件生成器
            async def event_generator():
                nonlocal docs_info_sent

                # 首先发送相关文档信息
                if relevant_docs and not docs_info_sent:
                    docs_info_event = {
                        "relevant_documents": relevant_docs,
                        "has_knowledge": True,
                        "status": "info"
                    }
                    yield f"data: {json.dumps(docs_info_event, ensure_ascii=False)}\n\n"
                    docs_info_sent = True

                # 然后发送生成的token
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.query(
            prompt=prompt,
            temperature=0.3,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        # 组装结果
        result.update({
            "relevant_documents": relevant_docs,
            "has_knowledge": True,
            "answer": result.get("content", "")  # 将内容映射为answer字段以保持一致性
        })

        return result

    except Exception as e:
        logger.error(f"基于知识库回答问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"基于知识库回答问题失败: {str(e)}")


# 按文档ID提问的新路由
@router.post("/knowledge-base/document/{document_id}/answer")
async def answer_from_specific_document(
        document_id: str = Path(..., description="文档ID"),
        question: str = Body(..., embed=True, description="问题"),
        stream: bool = Query(False, description="是否使用流式响应"),
        deepseek_service: DeepseekService = Depends(get_deepseek_service),
        document_service: DocumentService = Depends(get_document_service)
):
    """基于特定文档回答问题"""
    try:
        # 获取文档内容
        doc_content = await document_service.get_document_content(document_id)

        if doc_content is None:
            raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        # 获取文档元数据
        document = await document_service.get_document(document_id)
        document_info = None
        if document:
            document_info = {
                "document_id": document_id,
                "file_name": document.get("file_name"),
                "metadata": document.get("metadata", {}),
                "score": 1.0  # 手动指定的文档得分为1.0
            }

        # 构建提示
        prompt = f"""请根据提供的文档内容回答用户的问题。如果文档内容不足以完整回答问题，请明确指出，并仅基于文档中的信息提供部分回答。

文档内容:
{doc_content}

用户问题: {question}

请提供详细、准确、全面的回答，并引用文档中的信息支持您的回答。"""

        # 检查是否请求流式响应
        if stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.query(
                prompt=prompt,
                temperature=0.3,
                stream=True
            )

            # 添加文档信息
            doc_info_sent = False

            # 定义流式响应事件生成器
            async def event_generator():
                nonlocal doc_info_sent

                # 首先发送文档信息
                if document_info and not doc_info_sent:
                    doc_info_event = {
                        "relevant_documents": [document_info],
                        "has_knowledge": True,
                        "status": "info"
                    }
                    yield f"data: {json.dumps(doc_info_event, ensure_ascii=False)}\n\n"
                    doc_info_sent = True

                # 然后发送生成的token
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.query(
            prompt=prompt,
            temperature=0.3,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        # 组装结果
        result.update({
            "relevant_documents": [document_info] if document_info else [],
            "has_knowledge": True,
            "answer": result.get("content", "")  # 将内容映射为answer字段以保持一致性
        })

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"基于特定文档回答问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"基于特定文档回答问题失败: {str(e)}")

@router.post("/summarize")
async def summarize_text(
        request: SummarizeRequest,
        deepseek_service: DeepseekService = Depends(get_deepseek_service)
):
    """使用Deepseek总结文本内容，支持流式响应"""
    try:
        # 检查是否请求流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.summarize_document(
                text=request.text,
                max_length=request.max_length,
                stream=True
            )

            # 定义流式响应事件生成器
            async def event_generator():
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.summarize_document(
            text=request.text,
            max_length=request.max_length,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        return result
    except Exception as e:
        logger.error(f"生成摘要失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成摘要失败: {str(e)}")


@router.post("/key-points")
async def extract_key_points(
        request: KeyPointsRequest,
        deepseek_service: DeepseekService = Depends(get_deepseek_service)
):
    """使用Deepseek提取文本关键点，支持流式响应"""
    try:
        # 检查是否请求流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.extract_key_points(
                text=request.text,
                num_points=request.num_points,
                stream=True
            )

            # 定义流式响应事件生成器
            async def event_generator():
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.extract_key_points(
            text=request.text,
            num_points=request.num_points,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        return result# api/routers/deepseek.py (继续)
    except Exception as e:
        logger.error(f"提取关键点失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提取关键点失败: {str(e)}")


@router.post("/generate-questions")
async def generate_questions(
    request: GenerateQuestionsRequest,
    deepseek_service: DeepseekService = Depends(get_deepseek_service)
):
    """使用Deepseek生成问题，支持流式响应"""
    try:
        # 检查是否请求流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.generate_document_questions(
                text=request.text,
                num_questions=request.num_questions,
                stream=True
            )

            # 定义流式响应事件生成器
            async def event_generator():
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.generate_document_questions(
            text=request.text,
            num_questions=request.num_questions,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        return result
    except Exception as e:
        logger.error(f"生成问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成问题失败: {str(e)}")


@router.post("/sentiment")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    deepseek_service: DeepseekService = Depends(get_deepseek_service)
):
    """使用Deepseek分析文本情感，支持流式响应"""
    try:
        # 检查是否请求流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.analyze_sentiment(
                text=request.text,
                stream=True
            )

            # 定义流式响应事件生成器
            async def event_generator():
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.analyze_sentiment(
            text=request.text,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        return result
    except Exception as e:
        logger.error(f"情感分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"情感分析失败: {str(e)}")


@router.post("/compare")
async def compare_documents(
    request: CompareDocumentsRequest,
    deepseek_service: DeepseekService = Depends(get_deepseek_service)
):
    """使用Deepseek比较两个文档，支持流式响应"""
    try:
        # 检查是否请求流式响应
        if request.stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.compare_documents(
                doc1=request.doc1,
                doc2=request.doc2,
                stream=True
            )

            # 定义流式响应事件生成器
            async def event_generator():
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.compare_documents(
            doc1=request.doc1,
            doc2=request.doc2,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        return result
    except Exception as e:
        logger.error(f"文档比较失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档比较失败: {str(e)}")


@router.post("/document/{document_id}/summarize")
async def summarize_document(
    document_id: str = Path(..., description="文档ID"),
    max_length: int = Query(500, description="摘要最大长度"),
    stream: bool = Query(False, description="是否使用流式响应"),
    deepseek_service: DeepseekService = Depends(get_deepseek_service),
    document_service: DocumentService = Depends(get_document_service)
):
    """使用Deepseek总结特定文档，支持流式响应"""
    try:
        # 获取文档内容
        doc_content = await document_service.get_document_content(document_id)

        if doc_content is None:
            raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        # 获取文档元数据
        document = await document_service.get_document(document_id)
        document_info = None
        if document:
            document_info = {
                "document_id": document_id,
                "file_name": document.get("file_name"),
                "metadata": document.get("metadata", {})
            }

        # 检查是否请求流式响应
        if stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.summarize_document(
                text=doc_content,
                max_length=max_length,
                stream=True
            )

            # 添加文档信息
            doc_info_sent = False

            # 定义流式响应事件生成器
            async def event_generator():
                nonlocal doc_info_sent

                # 首先发送文档信息
                if document_info and not doc_info_sent:
                    doc_info_event = {
                        "document_info": document_info,
                        "status": "info"
                    }
                    yield f"data: {json.dumps(doc_info_event)}\n\n"
                    doc_info_sent = True

                # 然后发送生成的token
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.summarize_document(
            text=doc_content,
            max_length=max_length,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        # 添加文档信息
        if document_info:
            result["document_info"] = document_info

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档总结失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档总结失败: {str(e)}")


@router.post("/document/{document_id}/key-points")
async def extract_document_key_points(
    document_id: str = Path(..., description="文档ID"),
    num_points: int = Query(5, description="关键点数量"),
    stream: bool = Query(False, description="是否使用流式响应"),
    deepseek_service: DeepseekService = Depends(get_deepseek_service),
    document_service: DocumentService = Depends(get_document_service)
):
    """使用Deepseek提取特定文档的关键点，支持流式响应"""
    try:
        # 获取文档内容
        doc_content = await document_service.get_document_content(document_id)

        if doc_content is None:
            raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        # 获取文档元数据
        document = await document_service.get_document(document_id)
        document_info = None
        if document:
            document_info = {
                "document_id": document_id,
                "file_name": document.get("file_name"),
                "metadata": document.get("metadata", {})
            }

        # 检查是否请求流式响应
        if stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.extract_key_points(
                text=doc_content,
                num_points=num_points,
                stream=True
            )

            # 添加文档信息
            doc_info_sent = False

            # 定义流式响应事件生成器
            async def event_generator():
                nonlocal doc_info_sent

                # 首先发送文档信息
                if document_info and not doc_info_sent:
                    doc_info_event = {
                        "document_info": document_info,
                        "status": "info"
                    }
                    yield f"data: {json.dumps(doc_info_event)}\n\n"
                    doc_info_sent = True

                # 然后发送生成的token
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.extract_key_points(
            text=doc_content,
            num_points=num_points,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        # 添加文档信息
        if document_info:
            result["document_info"] = document_info

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档关键点提取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档关键点提取失败: {str(e)}")


@router.post("/document/{document_id}/answer")
async def answer_document_question(
    document_id: str = Path(..., description="文档ID"),
    question: str = Body(..., embed=True, description="问题"),
    stream: bool = Query(False, description="是否使用流式响应"),
    deepseek_service: DeepseekService = Depends(get_deepseek_service),
    document_service: DocumentService = Depends(get_document_service)
):
    """使用Deepseek回答关于特定文档的问题，支持流式响应"""
    try:
        # 获取文档内容
        doc_content = await document_service.get_document_content(document_id)

        if doc_content is None:
            raise HTTPException(status_code=404, detail=f"文档ID {document_id} 不存在")

        # 获取文档元数据
        document = await document_service.get_document(document_id)
        document_info = None
        if document:
            document_info = {
                "document_id": document_id,
                "file_name": document.get("file_name"),
                "metadata": document.get("metadata", {})
            }

        # 检查是否请求流式响应
        if stream:
            # 获取流式生成器
            stream_generator = await deepseek_service.answer_question(
                question=question,
                context=doc_content,
                stream=True
            )

            # 添加文档信息
            doc_info_sent = False

            # 定义流式响应事件生成器
            async def event_generator():
                nonlocal doc_info_sent

                # 首先发送文档信息
                if document_info and not doc_info_sent:
                    doc_info_event = {
                        "document_info": document_info,
                        "status": "info"
                    }
                    yield f"data: {json.dumps(doc_info_event)}\n\n"
                    doc_info_sent = True

                # 然后发送生成的token
                async for token_data in stream_generator:
                    if token_data.get("status") == "error":
                        yield f"data: {json.dumps(token_data)}\n\n"
                        break
                    yield f"data: {json.dumps(token_data)}\n\n"
                yield "data: [DONE]\n\n"

            # 返回SSE流式响应
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream"
            )

        # 非流式响应
        result = await deepseek_service.answer_question(
            question=question,
            context=doc_content,
            stream=False
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error", "调用Deepseek模型失败"))

        # 添加文档信息
        if document_info:
            result["document_info"] = document_info

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"回答文档问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"回答文档问题失败: {str(e)}")


# 添加新的辅助函数来清理文本
# 简单的清理函数
def clean_model_output(text: str) -> str:
    """清理模型输出中的特殊标记"""
    # 移除用户和助手标记
    text = re.sub(r'<[|\｜]User[|\｜]>|<[|\｜]Assistant[|\｜]>', '', text)

    # 移除思考过程
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    text = re.sub(r'<think>[\s\S]*', '', text)

    return text.strip()


