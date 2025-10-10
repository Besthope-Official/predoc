"""配置类工具"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    logger = None  # type: ignore


class BaseConfig(BaseModel):
    """配置基类，支持从 YAML 文件和环境变量加载配置。

    特性：
    - 默认从 `config.yaml` 读取，自动推断配置节名
    - 支持嵌套节（点分隔符，如 "models.chunking"）
    - YAML 不存在时回退到环境变量默认值
    """

    yaml_section: ClassVar[str | None] = None

    @classmethod
    def _default_config_path(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "config.yaml"

    @classmethod
    def _infer_section_name(cls) -> str:
        if cls.yaml_section:
            return cls.yaml_section
        name = cls.__name__
        name_lower = name.lower()
        if name_lower.endswith("config"):
            name_lower = name_lower[: -len("config")]
        return name_lower

    @classmethod
    def _filter_model_fields(cls, data: Mapping[str, Any] | None) -> dict:
        if not isinstance(data, Mapping):
            return {}
        allowed = set(getattr(cls, "model_fields", {}).keys())  # pydantic v2

        if not allowed:
            return dict(data)

        return {k: v for k, v in data.items() if k in allowed}

    @classmethod
    def from_yaml_dict(cls, path: str | None = None) -> dict:
        """从 YAML 文件加载配置字典。

        支持点分隔符表示嵌套节（如 "models.chunking"）。
        若文件不存在则返回空字典，允许使用环境变量回退。
        """
        cfg_path = Path(path) if path else cls._default_config_path()

        try:
            with open(cfg_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            if logger:
                logger.debug(
                    f"配置文件未找到: {cfg_path}. 将返回空配置以便使用环境变量/默认值。"
                )
            return {}
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件 YAML 解析失败: {e}") from e

        section = cls._infer_section_name()

        section_data = raw
        if section:
            for key in section.split("."):
                if isinstance(section_data, Mapping) and key in section_data:
                    section_data = section_data[key]
                else:
                    section_data = None
                    break

        if isinstance(section_data, Mapping):
            return cls._filter_model_fields(section_data)

        return cls._filter_model_fields(raw)

    @classmethod
    def from_yaml(cls, path: str | None = None) -> BaseConfig:
        """从 YAML 文件加载并实例化配置模型。

        若文件或必填字段缺失，将抛出 Pydantic 校验异常。
        """
        data = cls.from_yaml_dict(path)
        return cls(**data)
