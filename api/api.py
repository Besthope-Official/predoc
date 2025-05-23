'''封装 API'''

import os
import tempfile
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Body
from loguru import logger

from prep.processor import PDFProcessor
from .utils import ModelLoader, api_success, api_fail, ApiResponse


model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动，开始预加载模型...")
    model_loader.preload_all()

    yield

    logger.info("应用关闭，清理模型缓存...")
    model_loader.clear_cache()


app = FastAPI(lifespan=lifespan)


@app.post("/preprocess")
async def document_preprocess(
    docFile: UploadFile = File(...),
    docType: str = Form("article"),
    parseMethod: str = Form("auto"),
    chunkStrategy: str = Form("semantic"),
) -> ApiResponse:
    '''预处理文档接口'''
    if not docFile.filename:
        return api_fail(message="未提供文件")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, docFile.filename)

        try:
            with open(temp_file_path, "wb") as target_file:
                shutil.copyfileobj(docFile.file, target_file)
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            return api_fail(message="文件处理失败")
        finally:
            docFile.file.close()

        try:
            if docFile.filename.lower().endswith('.pdf'):
                chunker = model_loader.get_chunker(chunkStrategy)
                processor = PDFProcessor(
                    chunker=chunker,
                    parser=model_loader.parser,
                    embedder=model_loader.embedder,
                    output_dir=temp_dir,
                    upload_to_oss=True
                )

                result = processor.preprocess(temp_file_path)

            return api_success(data=result)
        except Exception as e:
            logger.error(f"预处理文档失败: {str(e)}")
            return api_fail(message=f"预处理错误: {str(e)}")


@app.post("/parser")
async def pdf_parse(
    file: UploadFile = File(...),
    parseMethod: str = Form("auto")
) -> ApiResponse:
    '''接收PDF文件和解析方式，返回解析后的文本'''
    if not file.filename.lower().endswith('.pdf'):
        return api_fail(message="只支持PDF文件")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        try:
            with open(temp_file_path, "wb") as pdf_file:
                shutil.copyfileobj(file.file, pdf_file)
        except Exception as e:
            logger.error(f"保存PDF文件失败: {str(e)}")
            return api_fail(message="文件处理失败")
        finally:
            file.file.close()

        try:
            text = model_loader.parser.parse(temp_file_path, temp_dir)
            return api_success(data={"text": text})
        except Exception as e:
            logger.error(f"PDF解析失败: {str(e)}")
            return api_fail(message=f"PDF解析错误: {str(e)}")


@app.post("/chunker")
async def document_chunk(
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
        return api_fail(message=f"分块错误: {str(e)}")


@app.post("/embedding")
async def text_embedding(
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
        return api_fail(message=f"嵌入生成错误: {str(e)}")
