"""AI 请求兼容性辅助逻辑。"""

import traceback
import threading
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional, Set


INPUT_TEXT_TYPE = "input_text"
INPUT_IMAGE_TYPE = "input_image"
IMAGE_DETAIL_AUTO = "auto"
JSON_OUTPUT_TYPE = "json_object"
UNSUPPORTED_JSON_OUTPUT_MARKERS = (
    "not supported by this model",
    "json_object",
    "json_schema",
    "text.format",
    "response_format.type",
)
# 请求参数错误的标记（部分中转站不支持某些可选参数）
UNSUPPORTED_PARAM_MARKERS = (
    "请求参数错误",
    "invalid parameter",
    "unrecognized parameter",
    "unknown parameter",
)
# 全局记录：已确认不被支持的可选参数名，后续请求自动跳过
_unsupported_params: Set[str] = set()
_unsupported_params_lock = threading.Lock()
# 可安全移除的可选参数列表
_OPTIONAL_PARAMS = ("temperature", "max_output_tokens")


def build_responses_input(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 Chat Completions 风格的消息转换为 Responses API 输入。"""
    input_items: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        input_items.append(
            {
                "role": role,
                "content": _build_input_content(message.get("content")),
            }
        )
    return input_items


def add_json_text_format(
    request_params: Dict[str, Any],
    enabled: bool,
) -> Dict[str, Any]:
    """按需附加 Responses API 的结构化 JSON 输出参数。"""
    next_params = dict(request_params)
    if not enabled:
        return next_params

    text_config = dict(next_params.get("text") or {})
    text_config["format"] = {"type": JSON_OUTPUT_TYPE}
    next_params["text"] = text_config
    return next_params


def is_json_output_unsupported_error(error: Exception) -> bool:
    """识别模型不支持结构化 JSON 输出参数的错误。"""
    message = str(error)
    return (
        "not supported" in message.lower()
        and any(marker in message for marker in UNSUPPORTED_JSON_OUTPUT_MARKERS)
    )


def is_param_unsupported_error(error: Exception) -> bool:
    """识别中转站不支持某些可选请求参数的错误（如 temperature）。

    对于中文标记 "请求参数错误"，该标记本身足够具体，无需错误信息中包含参数名
    （部分中转站返回空的参数名）。对于通用英文标记，仍要求同时包含可选参数名，
    避免误匹配其他类型的 "invalid parameter" 错误。
    """
    message = str(error).lower()
    has_marker = any(marker.lower() in message for marker in UNSUPPORTED_PARAM_MARKERS)
    if not has_marker:
        return False
    # "请求参数错误" 足够具体，中文 API 代理常省略参数名，单独匹配即可
    if "请求参数错误" in str(error):
        return True
    # 英文标记较通用，必须同时提到某个可选参数名，避免误匹配 model 等核心参数错误
    return any(p in message for p in _OPTIONAL_PARAMS)


def detect_unsupported_param_name(
    error: Exception, candidates: List[str]
) -> Optional[str]:
    """尝试从错误信息中识别具体不被支持的参数名。"""
    message = str(error).lower()
    for param in candidates:
        if param in message:
            return param
    return None


def strip_unsupported_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """移除已知不被支持的可选参数。"""
    with _unsupported_params_lock:
        blocked = set(_unsupported_params)
    if not blocked:
        return params
    return {k: v for k, v in params.items() if k not in blocked}


def mark_param_unsupported(param_name: str) -> None:
    """将某个参数标记为不支持，后续请求自动跳过。"""
    with _unsupported_params_lock:
        _unsupported_params.add(param_name)
    print(f"已标记参数 '{param_name}' 为不受支持，后续请求将自动跳过")


def reset_unsupported_params() -> None:
    """清除已记录的不支持参数（用于测试或重置状态）。"""
    with _unsupported_params_lock:
        _unsupported_params.clear()


def format_ai_error_detail(exc: Exception) -> str:
    """格式化 AI API 错误的详细信息，用于增强日志输出。

    提取错误类型、HTTP 状态码、API 错误 body 和完整堆栈。
    """
    parts = [f"[{type(exc).__name__}]: {exc}"]
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"  HTTP 状态码: {status_code}")
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(f"  API 错误详情: {body}")
    parts.append(f"  完整堆栈:\n{traceback.format_exc()}")
    return "\n".join(parts)


def get_optional_params_in(params: Dict[str, Any]) -> List[str]:
    """返回 params 中存在的可选参数名列表。"""
    return [p for p in _OPTIONAL_PARAMS if p in params]


def _log_request_params(params: Dict[str, Any]) -> None:
    """打印请求参数摘要，用于调试 API 调用。"""
    summary = {}
    for k, v in params.items():
        if k == "input":
            # input 可能含 base64 图片，只打印摘要
            if isinstance(v, list):
                summary[k] = f"[{len(v)} 条消息]"
            else:
                summary[k] = str(v)[:100] + "..."
        else:
            summary[k] = v
    print(f"[AI DEBUG] 请求参数: {summary}")


async def call_with_param_compat(
    create_fn: Callable[..., Coroutine],
    request_params: Dict[str, Any],
) -> Any:
    """调用 AI API，自动检测并移除不支持的可选参数后重试。

    Args:
        create_fn: 异步 API 调用函数（如 client.responses.create）
        request_params: 请求参数字典
    Returns:
        API 响应对象
    Raises:
        原始异常（当错误不是可选参数不支持时）
    """
    request_params = strip_unsupported_params(request_params)
    optional_in_request = get_optional_params_in(request_params)

    # 调试日志：打印请求参数（input 内容截断避免刷屏）
    _log_request_params(request_params)

    try:
        return await create_fn(**request_params)
    except Exception as exc:
        print(f"[AI DEBUG] is_param_unsupported_error={is_param_unsupported_error(exc)}, optional_in_request={optional_in_request}")
        if not (is_param_unsupported_error(exc) and optional_in_request):
            print(f"AI API 调用失败 {format_ai_error_detail(exc)}")
            raise
        # 尝试精确识别不支持的参数
        specific = detect_unsupported_param_name(exc, optional_in_request)
        if specific:
            mark_param_unsupported(specific)
        else:
            # 无法确定具体参数，移除所有可选参数
            for p in optional_in_request:
                mark_param_unsupported(p)
        request_params = strip_unsupported_params(request_params)
        print("检测到不支持的参数，已自动移除并重试")
        return await create_fn(**request_params)


def _build_input_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": INPUT_TEXT_TYPE, "text": content}]
    if not isinstance(content, list):
        raise ValueError(f"AI消息内容类型不受支持: {type(content).__name__}")

    return [_coerce_content_item(item) for item in content]


def _coerce_content_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"AI消息片段类型不受支持: {type(item).__name__}")

    item_type = item.get("type")
    if item_type in {"text", INPUT_TEXT_TYPE}:
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("文本消息片段缺少 text 字段。")
        return {"type": INPUT_TEXT_TYPE, "text": text}

    if item_type in {"image_url", INPUT_IMAGE_TYPE}:
        return _build_image_input_item(item)

    raise ValueError(f"不支持的 AI 消息片段类型: {item_type}")


def _build_image_input_item(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_image = item.get("image_url")
    if isinstance(raw_image, dict):
        image_url = raw_image.get("url")
    else:
        image_url = raw_image

    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("图片消息片段缺少有效的 image_url。")

    return {
        "type": INPUT_IMAGE_TYPE,
        "image_url": image_url,
        "detail": item.get("detail", IMAGE_DETAIL_AUTO),
    }
