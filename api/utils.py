'''封装 API'''
from typing import Any, Optional
from typing_extensions import TypedDict


class ApiResponse(TypedDict):
    """
    封装 API 的返回结果类型
    """
    success: bool
    data: Optional[Any]
    message: str


def api_response(success: bool, data: Any = None, message: str = "") -> ApiResponse:
    """
    封装 API 的返回结果

    :param success: 请求是否成功
    :param data: 返回的数据
    :param message: 错误或提示信息
    :return: 统一格式的字典
    """
    return {
        "success": success,
        "data": data,
        "message": message
    }


def api_success(data: Any = None, message: str = "Success") -> ApiResponse:
    """
    成功的 API 响应

    :param data: 返回的数据
    :param message: 提示信息
    :return: 成功响应格式的字典
    """
    return api_response(success=True, data=data, message=message)


def api_fail(message: str = "Fail", data: Any = None) -> ApiResponse:
    """
    失败的 API 响应

    :param message: 错误信息
    :param data: 相关的错误数据（如有）
    :return: 失败响应格式的字典
    """
    return api_response(success=False, data=data, message=message)
