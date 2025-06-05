'''封装 API'''

import os
import tempfile
import shutil
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Form, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import threading
from functools import lru_cache

from prep.processor import PDFProcessor
from .utils import ModelLoader, api_success, api_fail, ApiResponse
from task.task import TaskConsumer
from config.backend import RabbitMQConfig


@lru_cache()
def get_model_loader() -> ModelLoader:
    """获取模型加载器单例"""
    return ModelLoader()


@lru_cache()
def get_task_consumer() -> TaskConsumer:
    """获取任务消费者单例"""
    config = RabbitMQConfig()
    return TaskConsumer(config)


def start_task_consumer(consumer: TaskConsumer):
    """在后台线程中启动任务消费者"""
    try:
        consumer.start_consuming()
    except Exception as e:
        logger.error(f"启动任务消费者失败: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动，开始预加载模型...")
    model_loader = get_model_loader()
    model_loader.preload_all()

    logger.info("初始化任务消费者...")
    consumer = get_task_consumer()

    consumer_thread = threading.Thread(
        target=lambda: start_task_consumer(consumer),
        daemon=True
    )
    consumer_thread.start()

    yield

    logger.info("应用关闭，清理资源...")
    model_loader.clear_cache()


app = FastAPI(
    title="PreDoc API",
    description="文档预处理服务API",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=api_fail(message=str(exc.detail))
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=api_fail(message="服务器内部错误")
    )


async def get_processor(
    chunker_strategy: str,
    model_loader: Annotated[ModelLoader, Depends(get_model_loader)],
    temp_dir: str
) -> PDFProcessor:
    """获取文档处理器"""
    chunker = model_loader.get_chunker(chunker_strategy)
    return PDFProcessor(
        chunker=chunker,
        parser=model_loader.parser,
        embedder=model_loader.embedder,
        output_dir=temp_dir,
        upload_to_oss=True
    )


@app.post("/preprocess")
async def document_preprocess(
    model_loader: Annotated[ModelLoader, Depends(get_model_loader)],
    docFile: UploadFile = File(...),
    docType: str = Form("article"),
    parseMethod: str = Form("auto"),
    chunkStrategy: str = Form("semantic")
) -> ApiResponse:
    '''预处理文档接口'''
    if not docFile.filename:
        raise HTTPException(status_code=400, detail="未提供文件")

    if not docFile.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, docFile.filename)

        try:
            with open(temp_file_path, "wb") as target_file:
                shutil.copyfileobj(docFile.file, target_file)
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail="文件处理失败")
        finally:
            await docFile.close()

        try:
            processor = await get_processor(chunkStrategy, model_loader, temp_dir)
            result = processor.preprocess(temp_file_path)
            return api_success(data=result)
        except Exception as e:
            logger.error(f"预处理文档失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"预处理错误: {str(e)}")


@app.post("/parser")
async def pdf_parse(
    model_loader: Annotated[ModelLoader, Depends(get_model_loader)],
    file: UploadFile = File(...),
    parseMethod: str = Form("auto")
) -> ApiResponse:
    '''接收PDF文件和解析方式，返回解析后的文本'''
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(temp_file_path, "wb") as pdf_file:
                shutil.copyfileobj(file.file, pdf_file)
        except Exception as e:
            logger.error(f"保存PDF文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail="文件处理失败")
        finally:
            await file.close()

        try:
            text = model_loader.parser.parse(temp_file_path, temp_dir)
            return api_success(data={"text": text})
        except Exception as e:
            logger.error(f"PDF解析失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF解析错误: {str(e)}")


@app.post("/chunker")
async def document_chunk(
    model_loader: Annotated[ModelLoader, Depends(get_model_loader)],
    text: str = Body(...),
    chunkStrategy: str = Body("semantic")
) -> ApiResponse:
    '''接收一段文本，按照给定的分块方法，返回分块后的文本列表'''
    if not text.strip():
        logger.warning("收到空文本请求，返回空列表")
        return api_success(data=[])

    try:
        chunker = model_loader.get_chunker(chunkStrategy)
        chunk_result = chunker.chunk(text)
        return api_success(data=chunk_result)
    except Exception as e:
        logger.error(f"分块失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分块错误: {str(e)}")


@app.post("/embedding")
async def text_embedding(
    model_loader: Annotated[ModelLoader, Depends(get_model_loader)],
    text: str = Body(...),
    model: str = Body("paraphrase-multilingual-mpnet-base-v2")
) -> ApiResponse:
    '''对给定的文本，按照指定的嵌入模型，进行向量嵌入'''
    if not text.strip():
        logger.warning("收到空文本请求，返回空向量")
        return api_success(data={"embedding": []})

    try:
        embedding = model_loader.embedder.generate_embedding(text)
        logger.info(f"成功为文本生成嵌入向量，维度: {len(embedding)}")
        return api_success(data={"embedding": embedding.tolist()})
    except Exception as e:
        logger.error(f"生成嵌入向量失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"嵌入生成错误: {str(e)}")


@app.post("retrieval")
async def document_retrieval(
    query: str = Body(...),
    topK: int = Body(5),
) -> ApiResponse:
    '''接收查询字符串，返回检索到的文档列表'''
    pass
