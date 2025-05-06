import uvicorn
import argparse
from loguru import logger
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    parser = argparse.ArgumentParser(description="RAG API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="监听主机 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="监听端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (默认: 1)")
    args = parser.parse_args()

    logger.info(f"启动RAG API服务器于 {args.host}:{args.port}")
    logger.info(
        f"工作进程数: {args.workers}, 自动重载: {'启用' if args.reload else '禁用'}")

    uvicorn.run(
        "api.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
