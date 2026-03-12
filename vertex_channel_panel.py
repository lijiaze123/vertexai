"""
Vertex AI Gemini 渠道管理面板
基于 google-genai 库实现多渠道管理

部署：
    pip install fastapi uvicorn google-genai pillow
    python vertex_channel_panel.py

访问：http://localhost:9000/panel
"""

import base64
import io
import json
import os
import time
import logging
import uuid
import asyncio
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from PIL import Image
from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Vertex AI Gemini 渠道管理面板", version="1.0.0")

CONFIG_FILE = "data/vertex_channels.json"
REQUEST_HISTORY_FILE = "data/vertex_request_history.json"

# 请求历史记录（内存中保留最近100条）
request_history = []

def load_request_history():
    """从文件加载请求历史"""
    global request_history
    try:
        with open(REQUEST_HISTORY_FILE, 'r', encoding='utf-8') as f:
            request_history = json.load(f)
    except FileNotFoundError:
        request_history = []

def save_request_history():
    """保存请求历史到文件"""
    with open(REQUEST_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(request_history, f, indent=2, ensure_ascii=False)

# ==================== 数据模型 ====================

class ChannelConfig(BaseModel):
    name: str
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    location: str = "global"
    model: str = "gemini-2.5-flash-image-preview"
    enabled: bool = True
    priority: int = 10
    timeout: int = 300

class TestRequest(BaseModel):
    channel_id: str
    prompt: str = "一只可爱的猫咪"
    aspect_ratio: Optional[str] = None
    image_size: Optional[str] = "1K"
    image: Optional[str] = None  # base64编码的图片

class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = None
    image_size: Optional[str] = "1K"
    model: Optional[str] = None
    image: Optional[str] = None  # base64编码的图片

class GeminiRequest(BaseModel):
    contents: List[Any]
    generationConfig: Optional[Dict] = None

class OpenAIRequest(BaseModel):
    model: str
    messages: List[Dict]
    max_tokens: Optional[int] = None

class ApiKeyCreate(BaseModel):
    name: str
    mode: str  # "single" 或 "random"
    channel_id: Optional[str] = None  # 独立模式时指定渠道

# ==================== 配置管理 ====================

def load_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 确保有默认模型列表
        if "models" not in config:
            config["models"] = [
                "gemini-2.5-flash-image-preview",
                "gemini-3-pro-image-preview"
            ]
            save_config(config)
        return config
    except FileNotFoundError:
        default = {
            "channels": [],
            "models": [
                "gemini-2.5-flash-image-preview",
                "gemini-3-pro-image-preview"
            ]
        }
        save_config(default)
        return default

def save_config(config: Dict[str, Any]):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def get_enabled_channels() -> List[Dict]:
    config = load_config()
    channels = [c for c in config.get("channels", []) if c.get("enabled", False)]
    return sorted(channels, key=lambda x: x.get("priority", 999))

def get_api_keys() -> List[Dict]:
    """获取所有 API keys"""
    config = load_config()
    return config.get("api_keys", [])

def generate_api_key() -> str:
    """生成新的 API key"""
    return f"sk-{secrets.token_urlsafe(32)}"

def verify_api_key(api_key: str) -> Optional[Dict]:
    """验证 API key，返回 key 信息"""
    keys = get_api_keys()
    for k in keys:
        if k.get("key") == api_key and k.get("enabled", True):
            return k
    return None

def add_request_record(channel_name: str, model: str, prompt: str, aspect_ratio: str,
                       image_size: str, success: bool, elapsed: float, error: str = None):
    """添加请求记录"""
    global request_history
    # 使用北京时间 (UTC+8)
    beijing_tz = timezone(timedelta(hours=8))
    beijing_time = datetime.now(beijing_tz)
    record = {
        "time": beijing_time.strftime("%Y-%m-%d %H:%M:%S"),
        "channel": channel_name,
        "model": model,
        "prompt": prompt[:100],
        "aspect_ratio": aspect_ratio or "默认",
        "image_size": image_size,
        "status": "成功" if success else "失败",
        "elapsed": round(elapsed, 2),
        "error": error
    }
    request_history.insert(0, record)
    if len(request_history) > 100:
        request_history = request_history[:100]
    save_request_history()

# ==================== Vertex AI 调用 ====================

def create_gemini_client(channel: Dict) -> genai.Client:
    """创建 Gemini 客户端"""
    timeout_ms = channel.get('timeout', 300) * 1000
    http_options = HttpOptions(
        timeout=timeout_ms,
        retry_options=HttpRetryOptions(attempts=1, http_status_codes=())
    )

    api_key = channel.get('api_key')
    project_id = channel.get('project_id')
    location = channel.get('location', 'global')

    if api_key:
        return genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
    elif project_id:
        return genai.Client(vertexai=True, project=project_id, location=location, http_options=http_options)
    else:
        raise ValueError("必须提供 api_key 或 project_id")

def _safe_serialize(obj: Any) -> Any:
    if obj is None:
        return None
    for method in ("model_dump", "to_dict", "dict"):
        if hasattr(obj, method):
            try:
                return getattr(obj, method)()
            except Exception:
                pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)

def _truncate_strings(obj: Any, max_len: int = 2000) -> Any:
    if isinstance(obj, str):
        return obj[:max_len] + "...(truncated)" if len(obj) > max_len else obj
    if isinstance(obj, list):
        return [_truncate_strings(item, max_len) for item in obj]
    if isinstance(obj, dict):
        return {key: _truncate_strings(value, max_len) for key, value in obj.items()}
    return obj

def _collect_response_diagnostics(response: Any, candidate: Any = None) -> Dict[str, Any]:
    diagnostics = {
        "response_has_candidates": bool(getattr(response, "candidates", None)),
        "response_prompt_feedback": _safe_serialize(getattr(response, "prompt_feedback", None)),
        "response_model": getattr(response, "model", None),
    }
    if candidate is not None:
        diagnostics.update({
            "candidate_finish_reason": getattr(candidate, "finish_reason", None),
            "candidate_safety_ratings": _safe_serialize(getattr(candidate, "safety_ratings", None)),
            "candidate_content": _safe_serialize(getattr(candidate, "content", None)),
        })
    return _truncate_strings(diagnostics)

def _get_safety_block_reason(response: Any, candidate: Any = None) -> Optional[str]:
    reasons = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason and str(block_reason) not in ("BLOCK_REASON_UNSPECIFIED",):
            reasons.append(f"prompt_feedback.block_reason={block_reason}")
        block_message = getattr(prompt_feedback, "block_reason_message", None)
        if block_message:
            reasons.append(f"prompt_feedback.block_reason_message={block_message}")

    finish_reason = getattr(candidate, "finish_reason", None) if candidate is not None else None
    if finish_reason and str(finish_reason) in ("SAFETY", "BLOCKED"):
        reasons.append(f"candidate.finish_reason={finish_reason}")

    return "; ".join(reasons) if reasons else None

def call_gemini_api(channel: Dict, prompt: str, aspect_ratio: Optional[str] = None,
                    image_size: Optional[str] = None, image_base64: Optional[str] = None,
                    request_model: Optional[str] = None) -> Dict[str, Any]:
    """调用 Gemini API 生成图片"""
    start_time = time.time()

    try:
        client = create_gemini_client(channel)
        model = request_model or channel.get('model', 'gemini-2.5-flash-image-preview')

        # 构建内容
        contents = [prompt]
        if image_base64:
            # 将base64图片转换为PIL Image对象
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            contents = [image, prompt]

        # 构建配置
        config = None
        if aspect_ratio or image_size:
            from google.genai import types
            image_config_params = {}
            if aspect_ratio:
                image_config_params['aspect_ratio'] = aspect_ratio
            # gemini-3 系列模型支持 image_size 参数
            if image_size and (model.startswith('gemini-3') or model.startswith('gemini-2')):
                image_config_params['image_size'] = image_size

            if image_config_params:
                config = types.GenerateContentConfig(
                    image_config=types.ImageConfig(**image_config_params)
                )

        # 调用 API
        if config:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )

        # 提取图片数据
        if not response.candidates:
            logger.error("响应中没有 candidates")
            diagnostics = _collect_response_diagnostics(response)
            logger.error("响应诊断信息: %s", json.dumps(diagnostics, ensure_ascii=False))
            return {'success': False, 'error': '未返回有效响应'}

        candidate = response.candidates[0]

        # 检查 candidate.content 和 parts 是否存在
        if not hasattr(candidate, 'content') or candidate.content is None:
            logger.error("候选项中没有 content")
            diagnostics = _collect_response_diagnostics(response, candidate)
            logger.error("响应诊断信息: %s", json.dumps(diagnostics, ensure_ascii=False))
            block_reason = _get_safety_block_reason(response, candidate)
            if block_reason:
                return {'success': False, 'error': f'内容被安全过滤或拒绝生成：{block_reason}'}
            return {'success': False, 'error': '响应格式错误：缺少content'}

        if not hasattr(candidate.content, 'parts') or candidate.content.parts is None:
            logger.error("content 中没有 parts")
            diagnostics = _collect_response_diagnostics(response, candidate)
            logger.error("响应诊断信息: %s", json.dumps(diagnostics, ensure_ascii=False))
            block_reason = _get_safety_block_reason(response, candidate)
            if block_reason:
                return {'success': False, 'error': f'内容被安全过滤或拒绝生成：{block_reason}'}
            return {'success': False, 'error': '该内容可能违反政策，请修改提示词后重试'}

        logger.info(f"候选项数量: {len(response.candidates)}, parts数量: {len(candidate.content.parts)}")

        for i, part in enumerate(candidate.content.parts):
            logger.info(f"Part {i}: hasattr(inline_data)={hasattr(part, 'inline_data')}")
            logger.info(f"Part {i}: part内容={part}")
            if hasattr(part, 'inline_data') and part.inline_data is not None:
                logger.info(f"Part {i}: inline_data 不为空")
                image_data = part.inline_data.data
                if image_data:
                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size
                    image_base64 = base64.b64encode(image_data).decode('utf-8')

                    elapsed = time.time() - start_time
                    logger.info(f"成功生成图片: {width}x{height}, 耗时: {elapsed:.2f}s")
                    return {
                        'success': True,
                        'image_base64': image_base64,
                        'width': width,
                        'height': height,
                        'elapsed': elapsed
                    }
                else:
                    logger.warning(f"Part {i}: inline_data.data 为空")
            else:
                logger.warning(f"Part {i}: inline_data 为 None 或不存在")

        # 提取模型返回的文本信息（可能包含拒绝原因）
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

        error_message = '响应中没有图片数据'
        if text_parts:
            error_message = ' '.join(text_parts)

        logger.error(f"所有 parts 都没有有效的图片数据，模型返回: {error_message}")
        return {'success': False, 'error': error_message}

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        logger.error(f"调用失败: {error_msg}")
        return {'success': False, 'error': error_msg, 'elapsed': elapsed}

# ==================== API 端点 ====================

@app.get("/")
async def root():
    return {"message": "Vertex AI Gemini 渠道管理面板", "panel": "/panel"}

@app.get("/api/channels")
async def list_channels():
    config = load_config()
    return {"channels": config.get("channels", [])}

@app.post("/api/channels")
async def add_channel(channel: ChannelConfig):
    config = load_config()
    new_channel = channel.model_dump()
    new_channel['id'] = str(uuid.uuid4())
    new_channel['created_at'] = datetime.now().isoformat()
    config.setdefault("channels", []).append(new_channel)
    save_config(config)
    return {"success": True, "channel": new_channel}

@app.put("/api/channels/{channel_id}")
async def update_channel(channel_id: str, channel: ChannelConfig):
    config = load_config()
    for i, ch in enumerate(config.get("channels", [])):
        if ch.get("id") == channel_id:
            updated = channel.dict()
            updated['id'] = channel_id
            updated['created_at'] = ch.get('created_at')
            config["channels"][i] = updated
            save_config(config)
            return {"success": True, "channel": updated}
    raise HTTPException(status_code=404, detail="渠道不存在")

@app.delete("/api/channels/{channel_id}")
async def delete_channel(channel_id: str):
    config = load_config()
    channels = config.get("channels", [])
    config["channels"] = [ch for ch in channels if ch.get("id") != channel_id]
    save_config(config)
    return {"success": True}

@app.post("/api/test")
async def test_channel(req: TestRequest):
    config = load_config()
    channel = next((ch for ch in config.get("channels", []) if ch.get("id") == req.channel_id), None)
    if not channel:
        raise HTTPException(status_code=404, detail="渠道不存在")

    # 在线程池中执行同步调用，避免阻塞
    result = await asyncio.to_thread(call_gemini_api, channel, req.prompt, req.aspect_ratio, req.image_size, req.image)

    # 记录请求
    add_request_record(
        channel_name=channel.get('name'),
        model=channel.get('model'),
        prompt=req.prompt,
        aspect_ratio=req.aspect_ratio,
        image_size=req.image_size,
        success=result.get('success', False),
        elapsed=result.get('elapsed', 0),
        error=result.get('error')
    )

    return result

@app.post("/generate")
async def generate_image(req: GenerateRequest, authorization: Optional[str] = Header(None)):
    """图片生成接口（供外部调用）"""
    # 验证 API key
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]

    key_info = verify_api_key(api_key) if api_key else None
    if not key_info:
        raise HTTPException(status_code=401, detail="无效的 API Key")

    # 根据 API key 模式选择渠道
    if key_info.get("mode") == "single":
        # 独立模式：只使用指定的渠道
        channel_id = key_info.get("channel_id")
        config = load_config()
        channel = next((ch for ch in config.get("channels", []) if ch.get("id") == channel_id and ch.get("enabled")), None)
        if not channel:
            raise HTTPException(status_code=503, detail="指定的渠道不可用")
        channels = [channel]
    else:
        # 随机模式：从所有启用的渠道中选择
        channels = get_enabled_channels()

    if not channels:
        raise HTTPException(status_code=503, detail="没有可用的渠道")

    last_error = "所有渠道都失败了"
    for channel in channels:
        # 在线程池中执行同步调用，避免阻塞
        result = await asyncio.to_thread(call_gemini_api, channel, req.prompt, req.aspect_ratio, req.image_size, req.image, req.model)

        # 记录请求
        add_request_record(
            channel_name=channel.get('name'),
            model=req.model or channel.get('model'),
            prompt=req.prompt,
            aspect_ratio=req.aspect_ratio,
            image_size=req.image_size,
            success=result.get('success', False),
            elapsed=result.get('elapsed', 0),
            error=result.get('error')
        )

        if result.get('success'):
            return {
                "success": True,
                "image": f"data:image/png;base64,{result['image_base64']}",
                "width": result['width'],
                "height": result['height'],
                "channel": channel.get('name')
            }
        else:
            last_error = result.get('error', '未知错误')

    raise HTTPException(status_code=500, detail=last_error)

@app.post("/v1beta/models/{model}:generateContent")
async def gemini_generate(model: str, req: GeminiRequest, authorization: Optional[str] = Header(None)):
    """Gemini 格式的图片生成接口"""
    # 验证 API key
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]

    key_info = verify_api_key(api_key) if api_key else None
    if not key_info:
        raise HTTPException(status_code=401, detail="无效的 API Key")

    # 提取 prompt 和图片
    prompt = ""
    image_base64 = None
    if req.contents and len(req.contents) > 0:
        content = req.contents[0]
        if isinstance(content, dict) and "parts" in content:
            for part in content["parts"]:
                if isinstance(part, dict):
                    if "text" in part:
                        prompt = part["text"]
                    elif "inlineData" in part:
                        image_base64 = part["inlineData"].get("data")
        elif isinstance(content, str):
            prompt = content

    # 提取配置
    aspect_ratio = None
    image_size = "1K"
    if req.generationConfig:
        aspect_ratio = req.generationConfig.get("aspectRatio")
        image_size = req.generationConfig.get("imageSize", "1K")

    # 转换为内部请求格式
    internal_req = GenerateRequest(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        model=model
    )

    # 调用生成逻辑（复用 generate_image 的逻辑）
    if key_info.get("mode") == "single":
        channel_id = key_info.get("channel_id")
        config = load_config()
        channel = next((ch for ch in config.get("channels", []) if ch.get("id") == channel_id and ch.get("enabled")), None)
        if not channel:
            raise HTTPException(status_code=503, detail="指定的渠道不可用")
        channels = [channel]
    else:
        channels = get_enabled_channels()

    if not channels:
        raise HTTPException(status_code=503, detail="没有可用的渠道")

    last_error = "所有渠道都失败了"
    for channel in channels:
        result = await asyncio.to_thread(call_gemini_api, channel, internal_req.prompt, internal_req.aspect_ratio, internal_req.image_size, image_base64, model)

        add_request_record(
            channel_name=channel.get('name'),
            model=model or channel.get('model'),
            prompt=internal_req.prompt,
            aspect_ratio=internal_req.aspect_ratio,
            image_size=internal_req.image_size,
            success=result.get('success', False),
            elapsed=result.get('elapsed', 0),
            error=result.get('error')
        )

        if result.get('success'):
            return {
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": result['image_base64']
                            }
                        }]
                    }
                }]
            }
        else:
            last_error = result.get('error', '未知错误')

    raise HTTPException(status_code=500, detail=last_error)

@app.post("/v1/chat/completions")
async def openai_generate(req: OpenAIRequest, authorization: Optional[str] = Header(None)):
    """OpenAI 格式的图片生成接口"""
    # 验证 API key
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]

    key_info = verify_api_key(api_key) if api_key else None
    if not key_info:
        raise HTTPException(status_code=401, detail="无效的 API Key")

    # 提取 prompt 和图片（从最后一条用户消息）
    prompt = ""
    image_base64 = None
    for msg in reversed(req.messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            prompt = item.get("text", "")
                        elif item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image"):
                                image_base64 = image_url.split(",")[1]
            break

    # 转换为内部请求格式
    internal_req = GenerateRequest(
        prompt=prompt,
        model=req.model
    )

    # 调用生成逻辑
    if key_info.get("mode") == "single":
        channel_id = key_info.get("channel_id")
        config = load_config()
        channel = next((ch for ch in config.get("channels", []) if ch.get("id") == channel_id and ch.get("enabled")), None)
        if not channel:
            raise HTTPException(status_code=503, detail="指定的渠道不可用")
        channels = [channel]
    else:
        channels = get_enabled_channels()

    if not channels:
        raise HTTPException(status_code=503, detail="没有可用的渠道")

    last_error = "所有渠道都失败了"
    for channel in channels:
        result = await asyncio.to_thread(call_gemini_api, channel, internal_req.prompt, internal_req.aspect_ratio, internal_req.image_size, image_base64, req.model)

        add_request_record(
            channel_name=channel.get('name'),
            model=req.model or channel.get('model'),
            prompt=internal_req.prompt,
            aspect_ratio=internal_req.aspect_ratio,
            image_size=internal_req.image_size,
            success=result.get('success', False),
            elapsed=result.get('elapsed', 0),
            error=result.get('error')
        )

        if result.get('success'):
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"![Generated Image](data:image/png;base64,{result['image_base64']})"
                    },
                    "finish_reason": "stop"
                }]
            }
        else:
            last_error = result.get('error', '未知错误')

    raise HTTPException(status_code=500, detail=last_error)
async def get_logs():
    """获取最近的日志"""
    import subprocess
    try:
        result = subprocess.run(
            ['tail', '-100', '/tmp/claude-0/-data/tasks/b8acc59.output'],
            capture_output=True, text=True, timeout=5
        )
        return {"logs": result.stdout}
    except Exception as e:
        return {"logs": f"无法读取日志: {str(e)}"}

@app.get("/api/requests")
async def get_requests():
    """获取请求历史记录"""
    return {"requests": request_history}

@app.get("/api/models")
async def get_models():
    """获取模型列表"""
    config = load_config()
    return {"models": config.get("models", [])}

@app.post("/api/models")
async def add_model(model_name: str):
    """添加新模型"""
    config = load_config()
    models = config.get("models", [])
    if model_name not in models:
        models.append(model_name)
        config["models"] = models
        save_config(config)
    return {"success": True, "models": models}

@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """删除模型"""
    config = load_config()
    models = config.get("models", [])
    if model_name in models:
        models.remove(model_name)
        config["models"] = models
        save_config(config)
    return {"success": True, "models": models}

@app.get("/api/apikeys")
async def list_api_keys():
    """获取 API key 列表"""
    return {"api_keys": get_api_keys()}

@app.post("/api/apikeys")
async def create_api_key(req: ApiKeyCreate):
    """创建新的 API key"""
    config = load_config()
    new_key = {
        "id": str(uuid.uuid4()),
        "name": req.name,
        "key": generate_api_key(),
        "mode": req.mode,
        "channel_id": req.channel_id,
        "enabled": True,
        "created_at": datetime.now().isoformat()
    }
    config.setdefault("api_keys", []).append(new_key)
    save_config(config)
    return {"success": True, "api_key": new_key}

@app.delete("/api/apikeys/{key_id}")
async def delete_api_key(key_id: str):
    """删除 API key"""
    config = load_config()
    keys = config.get("api_keys", [])
    config["api_keys"] = [k for k in keys if k.get("id") != key_id]
    save_config(config)
    return {"success": True}

@app.get("/panel", response_class=HTMLResponse)
async def panel():
    # 异步读取文件，避免阻塞
    import aiofiles
    template_path = os.path.join(os.path.dirname(__file__), 'panel_template.html')
    async with aiofiles.open(template_path, 'r', encoding='utf-8') as f:
        return await f.read()

if __name__ == "__main__":
    import uvicorn
    load_request_history()
    uvicorn.run(app, host="0.0.0.0", port=9000)
