from milvus.milvus import store_embedding_task
from random import random
from models import Task

def preprocess(task: Task):
    """
    进行预处理任务，从 OSS 上获取文件，并进行预处理
    预处理后得到的图表会上传到 OSS 上
    分块后嵌入的文本直接存储到 Milvus 中
    """
    import time
    time.sleep(20)
    store_embedding_task(
        [[random() for _ in range(768)] for _ in range(10)],
        ["chunk test" for _ in range(10)],
        task
    )
    
