import os
import shutil
import sys
from pathlib import Path


def clean_output_directory():
    """清除 output 目录中的所有文件和子目录"""

    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent
    output_dir = project_root / 'output'

    if not output_dir.exists():
        print(f"输出目录不存在: {output_dir}")
        return

    user_confirm = input("are u sure? (y/n): ")
    if user_confirm.lower() not in ["y", "yes"]:
        print("操作已取消")
        return

    try:
        item_count = sum(1 for _ in output_dir.iterdir())

        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
                print(f"已删除文件: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"已删除目录: {item.name}")

        print(f"清理完成：共删除了 {item_count} 个项目")

    except Exception as e:
        print(f"清理时发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("开始清理 output 目录...")
    clean_output_directory()
