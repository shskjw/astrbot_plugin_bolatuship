"""
API 管理器 - 处理视频生成 API 调用

支持:
- 柏拉图 API (原生接口)
- OpenAI 兼容格式 API (Chat Completions)
- OpenAI Videos API (直接视频生成)
- Runway/Pika 等第三方 API 格式
"""

import asyncio
import base64
import json
import re
import time
import aiohttp
from typing import List, Dict, Optional, Tuple, Any
from astrbot.api import logger


class ApiManager:
    """API 管理器 - 支持多种视频生成 API 格式"""
    
    def __init__(self, config: dict):
        self.config = config
        self.key_lock = asyncio.Lock()
        self.plato_key_index = 0
        self.openai_key_index = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP Session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """关闭 Session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_plato_api_key(self) -> Optional[str]:
        """轮询获取柏拉图 API Key"""
        async with self.key_lock:
            keys = self.config.get("plato_api_keys", [])
            if not keys:
                return None
            key = keys[self.plato_key_index % len(keys)]
            self.plato_key_index += 1
            return key

    async def get_openai_api_key(self) -> Optional[str]:
        """轮询获取 OpenAI API Key"""
        async with self.key_lock:
            keys = self.config.get("openai_api_keys", [])
            if not keys:
                return None
            key = keys[self.openai_key_index % len(keys)]
            self.openai_key_index += 1
            return key

    def get_mime_type(self, data: bytes) -> str:
        """检测图片 MIME 类型"""
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif data.startswith(b'\xff\xd8'):
            return 'image/jpeg'
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return 'image/gif'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            return 'image/webp'
        return 'image/png'

    # ================= 柏拉图原生 API =================

    async def submit_plato_task(
        self,
        prompt: str,
        params: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        proxy: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        提交柏拉图视频生成任务
        
        Args:
            prompt: 提示词
            params: 额外参数 (aspect_ratio, hd, duration 等)
            image_bytes: 图片数据 (图生视频时使用)
            proxy: 代理地址
            
        Returns:
            (task_id, error_message)
        """
        api_url = self.config.get("api_url", "https://api.bltcy.ai")
        api_key = await self.get_plato_api_key()
        if not api_key:
            return None, "无可用的柏拉图 API Key"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 使用配置的模型
        model = self.config.get("default_model", "sora-2")

        payload = {
            "prompt": prompt,
            "model": model,
            **params
        }

        # 图生视频：添加图片
        if image_bytes:
            mime = self.get_mime_type(image_bytes)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            payload["images"] = [f"data:{mime};base64,{base64_image}"]

        try:
            timeout = aiohttp.ClientTimeout(total=180)
            session = await self._get_session()
            # 使用 URL 构建方法自动拼接路径
            endpoint = self._build_api_url(api_url, "plato")
            logger.debug(f"[VideoGen] 柏拉图 API 端点: {endpoint}")
            
            async with session.post(endpoint, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_info = data.get('error', {}).get('message', str(data))
                    return None, f"任务提交失败 ({resp.status}): {error_info}"

                task_id = data.get("task_id")
                if not task_id:
                    # 尝试其他字段
                    task_id = data.get("id") or data.get("generation_id") or data.get("job_id")
                if not task_id:
                    return None, f"未能从响应中获取 task_id: {json.dumps(data)}"
                return task_id, "提交成功"
                
        except asyncio.TimeoutError:
            return None, "请求API超时，服务器可能正忙，请稍后再试"
        except Exception as e:
            logger.error(f"[VideoGen] 任务提交网络错误: {e}", exc_info=True)
            return None, f"网络错误: {e}"

    async def poll_plato_result(
        self,
        task_id: str,
        proxy: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        轮询柏拉图任务结果
        
        Args:
            task_id: 任务ID
            proxy: 代理地址
            
        Returns:
            (video_url, status_message)
        """
        api_key = await self.get_plato_api_key()
        if not api_key:
            return None, "无可用的柏拉图 API Key"

        api_url = self.config.get("api_url", "https://api.bltcy.ai")
        timeout_seconds = self.config.get("polling_timeout", 300)
        interval = self.config.get("polling_interval", 5)
        
        start_time = time.monotonic()

        headers = {"Authorization": f"Bearer {api_key}"}
        # 构建轮询端点 URL
        base_endpoint = self._build_api_url(api_url, "plato")
        endpoint = f"{base_endpoint}/{task_id}"
        logger.debug(f"[VideoGen] 轮询端点: {endpoint}")
        session = await self._get_session()

        while time.monotonic() - start_time < timeout_seconds:
            try:
                async with session.get(endpoint, headers=headers, proxy=proxy, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get("status", "").upper()

                        if status in ["SUCCESS", "COMPLETED", "DONE", "FINISHED"]:
                            video_url = self._extract_video_url_from_result(data)
                            if video_url:
                                return video_url, "生成成功"
                            else:
                                logger.error(f"成功但未找到视频链接: {json.dumps(data)}")
                                return None, "任务成功但响应中未找到视频链接"

                        elif status in ["FAILURE", "FAILED", "ERROR"]:
                            reason = data.get("fail_reason") or data.get("error") or data.get("message") or "未知错误"
                            if isinstance(reason, dict):
                                reason = reason.get("message", str(reason))
                            try:
                                reason_json = json.loads(reason) if isinstance(reason, str) else reason
                                reason = reason_json.get("message", str(reason))
                            except:
                                pass
                            return None, f"任务失败: {reason}"

                        elif status in ["PENDING", "PROCESSING", "RUNNING", "QUEUED", "IN_PROGRESS"]:
                            # 继续等待
                            pass
                        else:
                            # 未知状态，检查是否有结果
                            video_url = self._extract_video_url_from_result(data)
                            if video_url:
                                return video_url, "生成成功"

                    else:
                        logger.warning(f"轮询状态码异常: {resp.status}, 响应: {await resp.text()}")

            except Exception as e:
                logger.warning(f"轮询状态时发生网络异常: {e}")

            await asyncio.sleep(interval)

        return None, "任务超时"

    def _extract_video_url_from_result(self, data: Dict) -> Optional[str]:
        """从轮询结果中提取视频 URL"""
        # 尝试多种可能的字段
        possible_fields = [
            ("data", "output"),
            ("data", "video_url"),
            ("data", "url"),
            ("result", "video_url"),
            ("result", "url"),
            ("result", "output"),
            ("output",),
            ("video_url",),
            ("url",),
            ("video",),
            ("generation", "video_url"),
            ("generation", "url"),
        ]
        
        for fields in possible_fields:
            value = data
            for field in fields:
                if isinstance(value, dict) and field in value:
                    value = value[field]
                else:
                    value = None
                    break
            if value and isinstance(value, str) and (value.startswith("http") or value.startswith("data:")):
                return value
        
        # 尝试在整个响应中搜索 URL
        data_str = json.dumps(data)
        url_match = re.search(r'(https?://[^\s<>")\]\\]+\.(?:mp4|webm|mov|avi|m3u8))', data_str, re.IGNORECASE)
        if url_match:
            return url_match.group(1)
        
        return None

    # ================= OpenAI 兼容格式 API =================

    async def call_openai_video_api(
        self,
        prompt: str,
        params: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        proxy: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        调用 OpenAI 兼容格式的视频生成 API
        
        支持多种格式:
        - Chat Completions 格式
        - Videos/Generations 格式
        - 异步任务格式
        
        Args:
            prompt: 提示词
            params: 额外参数
            image_bytes: 图片数据
            proxy: 代理地址
            
        Returns:
            (video_url, error_message)
        """
        api_url = self.config.get("openai_api_url", "")
        if not api_url:
            return None, "未配置 OpenAI 兼容 API URL"
            
        api_key = await self.get_openai_api_key()
        if not api_key:
            return None, "无可用的 OpenAI API Key"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        model = self.config.get("openai_model", "sora-2")
        
        # 根据 URL 判断 API 类型
        if "videos/generations" in api_url.lower() or "video/generations" in api_url.lower():
            return await self._call_videos_api(api_url, headers, model, prompt, params, image_bytes, proxy)
        else:
            return await self._call_chat_completions_api(api_url, headers, model, prompt, params, image_bytes, proxy)

    async def _call_videos_api(
        self,
        api_url: str,
        headers: Dict[str, str],
        model: str,
        prompt: str,
        params: Dict[str, Any],
        image_bytes: Optional[bytes],
        proxy: Optional[str]
    ) -> Tuple[Optional[str], str]:
        """调用 Videos/Generations 格式 API"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            **params
        }
        
        # 添加图片
        if image_bytes:
            mime = self.get_mime_type(image_bytes)
            b64 = base64.b64encode(image_bytes).decode()
            payload["image"] = f"data:{mime};base64,{b64}"
            # 某些 API 使用 images 数组
            payload["images"] = [f"data:{mime};base64,{b64}"]

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 300))
            session = await self._get_session()
            
            async with session.post(api_url, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
                resp_text = await resp.text()
                
                if resp.status not in [200, 201, 202]:
                    return None, self._parse_error_response(resp.status, resp_text)

                try:
                    res_data = json.loads(resp_text)
                except json.JSONDecodeError:
                    return None, f"数据解析失败: 返回内容不是 JSON"

                if "error" in res_data:
                    return None, json.dumps(res_data["error"], ensure_ascii=False)

                # 检查是否是异步任务
                task_id = res_data.get("id") or res_data.get("task_id") or res_data.get("generation_id")
                status = res_data.get("status", "").upper()
                
                if task_id and status in ["PENDING", "PROCESSING", "QUEUED", "IN_PROGRESS", ""]:
                    # 异步任务，需要轮询
                    return await self._poll_openai_task(api_url, headers, task_id, proxy)

                # 同步返回结果
                video_url = self._extract_video_url(res_data)
                if video_url:
                    return video_url, "生成成功"
                    
                return None, f"API请求成功但未找到视频数据。Raw: {str(res_data)[:200]}..."

        except asyncio.TimeoutError:
            return None, f"请求超时，请稍后再试"
        except Exception as e:
            logger.error(f"[VideoGen] Videos API 调用错误: {e}", exc_info=True)
            return None, f"系统错误: {e}"

    async def _call_chat_completions_api(
        self,
        api_url: str,
        headers: Dict[str, str],
        model: str,
        prompt: str,
        params: Dict[str, Any],
        image_bytes: Optional[bytes],
        proxy: Optional[str]
    ) -> Tuple[Optional[str], str]:
        """调用 Chat Completions 格式 API"""
        
        # 构建消息内容
        content_list = [{"type": "text", "text": prompt}]
        
        if image_bytes:
            mime = self.get_mime_type(image_bytes)
            b64 = base64.b64encode(image_bytes).decode()
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_list}],
            "stream": False,
        }
        
        # 添加额外参数
        for key, value in params.items():
            if key not in payload:
                payload[key] = value

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 300))
            session = await self._get_session()
            
            async with session.post(api_url, json=payload, headers=headers, proxy=proxy, timeout=timeout) as resp:
                resp_text = await resp.text()
                
                if resp.status not in [200, 201]:
                    # 检查是否是不支持 chat completions 的错误
                    if self._is_chat_not_supported_error(resp_text):
                        # 尝试转换为 videos API
                        videos_url = self._convert_to_videos_api_url(api_url)
                        logger.info(f"[VideoGen] Chat API 不支持，尝试 Videos API: {videos_url}")
                        return await self._call_videos_api(videos_url, headers, model, prompt, params, image_bytes, proxy)
                    
                    return None, self._parse_error_response(resp.status, resp_text)

                try:
                    res_data = json.loads(resp_text)
                except json.JSONDecodeError:
                    # 尝试解析流式响应
                    res_data = self._parse_stream_response(resp_text)
                    if not res_data:
                        return None, f"数据解析失败: 返回内容不是 JSON"

                if "error" in res_data:
                    return None, json.dumps(res_data["error"], ensure_ascii=False)

                # 解析视频 URL
                video_url, error_msg = self._extract_video_url_with_error(res_data)
                if video_url:
                    return video_url, "生成成功"
                
                # 如果有错误消息，返回错误
                if error_msg:
                    return None, error_msg
                    
                return None, f"API请求成功但未找到视频数据。Raw: {str(res_data)[:200]}..."

        except asyncio.TimeoutError:
            return None, f"请求超时，请稍后再试"
        except Exception as e:
            logger.error(f"[VideoGen] Chat API 调用错误: {e}", exc_info=True)
            return None, f"系统错误: {e}"

    async def _poll_openai_task(
        self,
        base_url: str,
        headers: Dict[str, str],
        task_id: str,
        proxy: Optional[str]
    ) -> Tuple[Optional[str], str]:
        """轮询 OpenAI 格式的异步任务"""
        
        timeout_seconds = self.config.get("polling_timeout", 300)
        interval = self.config.get("polling_interval", 5)
        start_time = time.monotonic()
        
        # 构建轮询 URL
        poll_url = f"{base_url.rstrip('/')}/{task_id}"
        if not poll_url.endswith(task_id):
            poll_url = base_url.replace("/generations", f"/generations/{task_id}")
        
        session = await self._get_session()
        
        while time.monotonic() - start_time < timeout_seconds:
            try:
                async with session.get(poll_url, headers=headers, proxy=proxy, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get("status", "").upper()
                        
                        if status in ["SUCCESS", "COMPLETED", "DONE", "FINISHED", "SUCCEEDED"]:
                            video_url = self._extract_video_url(data)
                            if video_url:
                                return video_url, "生成成功"
                            return None, "任务成功但未找到视频链接"
                        
                        elif status in ["FAILURE", "FAILED", "ERROR"]:
                            error = data.get("error") or data.get("message") or "未知错误"
                            return None, f"任务失败: {error}"
                        
                        # 继续等待
                    else:
                        logger.warning(f"轮询状态码异常: {resp.status}")
                        
            except Exception as e:
                logger.warning(f"轮询异常: {e}")
            
            await asyncio.sleep(interval)
        
        return None, "任务超时"

    def _extract_video_url_with_error(self, data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        从 API 响应中提取视频 URL，同时检测错误消息
        
        Returns:
            (video_url, error_message) - 如果是错误则返回 (None, error_msg)
        """
        try:
            # 检查 Chat Completion 格式中的错误消息
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                if content:
                    # 检查是否是明确的错误消息
                    if self._is_error_content(content):
                        error_msg = content[:200]
                        if len(content) > 200:
                            error_msg += "..."
                        return None, f"API 返回错误: {error_msg}"
                    
                    # 尝试提取视频 URL
                    video_url = self._extract_video_url(data)
                    if video_url:
                        return video_url, None
                    
                    # 没有找到 URL，检查是否是内容审核问题
                    moderation_error = self._check_content_moderation(content)
                    if moderation_error:
                        return None, moderation_error
            
            # 尝试从其他位置提取视频 URL
            video_url = self._extract_video_url(data)
            return video_url, None
            
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return None, f"解析响应失败: {e}"

    def _extract_video_url(self, data: Dict) -> Optional[str]:
        """从 API 响应中提取视频 URL（增强版）"""
        try:
            # 1. 标准 OpenAI Videos 格式: {"data": [{"url": "..."}]}
            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                item = data["data"][0]
                for key in ["url", "video_url", "video", "output", "result"]:
                    if key in item and item[key]:
                        return item[key]
                # 检查 b64_json
                if "b64_json" in item:
                    return f"data:video/mp4;base64,{item['b64_json']}"

            # 2. Chat Completion 格式: {"choices": [{"message": {"content": "..."}}]}
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})
                
                # 检查直接的 video_url 字段
                for key in ["video_url", "video", "url", "output"]:
                    if key in message and message[key]:
                        return message[key]
                
                # 检查 tool_calls
                if "tool_calls" in message and isinstance(message["tool_calls"], list):
                    for tool in message["tool_calls"]:
                        func_args = tool.get("function", {}).get("arguments", "")
                        url = self._extract_url_from_text(func_args)
                        if url:
                            return url
                
                # 检查 content 中的 URL
                content = message.get("content", "")
                if content:
                    # 先检查是否是错误消息
                    if self._is_error_content(content):
                        return None
                    url = self._extract_url_from_text(content)
                    if url:
                        return url

            # 3. 直接字段
            for key in ["video_url", "url", "output", "video", "result", "generation_url"]:
                if key in data and data[key]:
                    value = data[key]
                    if isinstance(value, str) and (value.startswith("http") or value.startswith("data:")):
                        return value
                    elif isinstance(value, dict):
                        for subkey in ["url", "video_url", "output"]:
                            if subkey in value and value[subkey]:
                                return value[subkey]

            # 4. 嵌套结构
            for parent_key in ["result", "data", "output", "generation"]:
                if parent_key in data and isinstance(data[parent_key], dict):
                    nested = data[parent_key]
                    for key in ["video_url", "url", "output", "video"]:
                        if key in nested and nested[key]:
                            return nested[key]

            # 5. 在整个响应中搜索视频 URL
            data_str = json.dumps(data)
            url = self._extract_url_from_text(data_str)
            if url:
                return url

        except Exception as e:
            logger.error(f"解析视频 URL 失败: {e}")
        
        return None

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """从文本中提取视频 URL"""
        if not text:
            return None
        
        # 匹配视频文件 URL（优先级最高）
        video_match = re.search(r'(https?://[^\s<>")\]\\]+\.(?:mp4|webm|mov|avi|m3u8|mkv))', text, re.IGNORECASE)
        if video_match:
            return video_match.group(1).rstrip(")>,'\".")
        
        # 匹配 Markdown 格式 [text](url) - 常见于 AI 生成的响应
        md_match = re.search(r'\[.*?\]\((https?://[^\s<>")\]]+)\)', text)
        if md_match:
            url = md_match.group(1).rstrip(")>,'\".")
            return url  # Markdown 格式的 URL 通常是有效的
        
        # 匹配 "video" 或 "url" 后面的链接
        video_url_match = re.search(r'(?:video|url|link|视频|链接)[:\s]*["\']?(https?://[^\s<>"\')\]\\]+)', text, re.IGNORECASE)
        if video_url_match:
            return video_url_match.group(1).rstrip(")>,'\".")
        
        # 匹配 JSON 格式中的 URL
        json_url_match = re.search(r'"(?:url|video_url|video|output|result)":\s*"(https?://[^"]+)"', text)
        if json_url_match:
            return json_url_match.group(1)
        
        # 匹配 "Here is" 或 "here's" 后面的链接（常见于 AI 响应）
        here_match = re.search(r'(?:here\s*(?:is|\'s)|download|watch|view)[:\s]*["\']?(https?://[^\s<>"\')\]\\]+)', text, re.IGNORECASE)
        if here_match:
            return here_match.group(1).rstrip(")>,'\".")
        
        # 匹配 "I generated" 或 "generated a video" 后面的链接
        generated_match = re.search(r'(?:generated|created|made)\s+(?:a\s+)?(?:video|clip)[^:]*[:\s]*(https?://[^\s<>"\')\]\\]+)', text, re.IGNORECASE)
        if generated_match:
            return generated_match.group(1).rstrip(")>,'\".")
        
        # 匹配通用 URL（需要看起来像视频或媒体链接）
        url_match = re.search(r'(https?://[^\s<>")\]\\]+)', text)
        if url_match:
            url = url_match.group(1).rstrip(")>,'\".")
            if self._looks_like_video_url(url):
                return url
        
        # 匹配 base64 视频数据
        b64_match = re.search(r'(data:video/[a-zA-Z0-9]+;base64,[a-zA-Z0-9+/=]+)', text)
        if b64_match:
            return b64_match.group(1)
        
        return None

    def _looks_like_video_url(self, url: str) -> bool:
        """判断 URL 是否看起来像视频链接"""
        url_lower = url.lower()
        # 视频文件扩展名
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.m3u8', '.mkv', '.flv']
        if any(ext in url_lower for ext in video_extensions):
            return True
        # 常见视频服务关键词
        video_keywords = ['video', 'media', 'stream', 'cdn', 'generation', 'output', 'result']
        if any(kw in url_lower for kw in video_keywords):
            return True
        return False

    def _is_error_content(self, content: str) -> bool:
        """检查内容是否是错误消息而非视频结果"""
        if not content:
            return False
        
        content_lower = content.lower()
        
        # 常见错误关键词
        error_keywords = [
            "not enough requests",
            "rate limit",
            "quota exceeded",
            "insufficient",
            "please wait",
            "upgrade your",
            "error:",
            "failed:",
            "unable to",
            "cannot process",
            "invalid request",
            "unauthorized",
            "forbidden",
            "too many requests",
            "service unavailable",
            "internal server error",
            "bad request",
            "content policy",
            "safety",
            "inappropriate",
            "violates",
            "not allowed",
            "blocked",
            "rejected",
            "请求过多",
            "配额不足",
            "请稍后",
            "升级",
            "限制",
            "错误",
            "失败",
            "违规",
            "敏感",
            "审核",
            "不允许",
            "拒绝",
        ]
        
        return any(kw in content_lower for kw in error_keywords)

    def _check_content_moderation(self, content: str) -> Optional[str]:
        """检查是否是内容审核导致的问题"""
        if not content:
            return None
        
        content_lower = content.lower()
        
        # 检查是否显示生成完成但没有 URL
        has_progress_100 = "100%" in content or "进度100" in content
        has_generated = "generated" in content_lower or "生成" in content
        has_url = "http" in content_lower or "https" in content_lower
        
        if (has_progress_100 or has_generated) and not has_url:
            # 检查是否有内容审核相关的提示
            moderation_keywords = [
                "content policy", "safety", "inappropriate", "violates",
                "not allowed", "blocked", "rejected", "moderation",
                "违规", "敏感", "审核", "不允许", "拒绝", "限制"
            ]
            
            for kw in moderation_keywords:
                if kw in content_lower:
                    return f"内容可能触发了审核限制: {kw}"
            
            # 没有明确的审核提示，但生成完成却没有 URL
            return "视频生成完成但未返回链接，可能是内容审核限制或 API 返回格式异常"
        
        return None

    def _parse_error_response(self, status: int, resp_text: str) -> str:
        """解析错误响应"""
        try:
            err_json = json.loads(resp_text)
            if "error" in err_json:
                error = err_json["error"]
                if isinstance(error, dict):
                    return f"API Error {status}: {error.get('message', str(error))}"
                return f"API Error {status}: {error}"
            if "message" in err_json:
                return f"API Error {status}: {err_json['message']}"
        except:
            pass
        
        if "<html" in resp_text.lower():
            return f"HTTP {status}: 服务端返回了网页而非数据，请检查URL配置"
        
        return f"HTTP {status}: {resp_text[:200]}"

    def _is_chat_not_supported_error(self, error_text: str) -> bool:
        """检查是否是 chat completions 不支持的错误"""
        error_lower = error_text.lower()
        keywords = [
            "does not support chat completions",
            "not support chat",
            "chat completions not supported",
            "use videos api",
            "videos/generations",
            "not a chat model",
            "video generation model",
            "invalid model for chat"
        ]
        return any(kw in error_lower for kw in keywords)

    def _normalize_api_url(self, url: str, endpoint_type: str = "videos") -> str:
        """
        标准化 API URL，自动补全路径
        
        Args:
            url: 原始 URL
            endpoint_type: 端点类型 (videos, chat, images)
            
        Returns:
            标准化后的完整 URL
        """
        url = url.strip().rstrip("/")
        
        # 如果 URL 已经包含完整路径，直接返回
        if endpoint_type == "videos":
            if "videos/generations" in url.lower() or "video/generations" in url.lower():
                return url
        elif endpoint_type == "chat":
            if "chat/completions" in url.lower():
                return url
        elif endpoint_type == "images":
            if "images/generations" in url.lower():
                return url
        
        # 根据端点类型构建路径
        endpoint_paths = {
            "videos": "/videos/generations",
            "chat": "/chat/completions",
            "images": "/images/generations"
        }
        endpoint_path = endpoint_paths.get(endpoint_type, "/videos/generations")
        
        # 检查是否已有版本号
        if "/v1" in url or "/v2" in url:
            # 找到版本号位置
            for version in ["/v1", "/v2", "/v3"]:
                if version in url:
                    idx = url.find(version) + len(version)
                    base = url[:idx]
                    # 检查版本号后面是否已有路径
                    remaining = url[idx:]
                    if remaining and not remaining.startswith("/"):
                        remaining = "/" + remaining
                    # 如果后面没有有效路径，添加端点路径
                    if not remaining or remaining == "/":
                        return f"{base}{endpoint_path}"
                    # 如果后面有其他路径但不是目标端点，替换
                    return f"{base}{endpoint_path}"
        
        # 没有版本号，添加 /v1 和端点路径
        return f"{url}/v1{endpoint_path}"

    def _convert_to_videos_api_url(self, chat_url: str) -> str:
        """将 chat/completions URL 转换为 videos/generations URL"""
        url = chat_url.rstrip("/")
        
        # 直接替换路径
        if "chat/completions" in url:
            return url.replace("chat/completions", "videos/generations")
        
        # 使用标准化方法
        return self._normalize_api_url(url, "videos")

    def _build_api_url(self, base_url: str, api_type: str = "plato") -> str:
        """
        根据 API 类型构建完整的 API URL
        
        Args:
            base_url: 基础 URL
            api_type: API 类型 (plato, openai_videos, openai_chat)
            
        Returns:
            完整的 API URL
        """
        base_url = base_url.strip().rstrip("/")
        
        if api_type == "plato":
            # 柏拉图 API
            if "/v2/videos/generations" in base_url:
                return base_url
            if "/v1/videos/generations" in base_url:
                return base_url
            if base_url.endswith("/v2") or base_url.endswith("/v1"):
                return f"{base_url}/videos/generations"
            # 默认使用 v2
            return f"{base_url}/v2/videos/generations"
        
        elif api_type == "openai_videos":
            return self._normalize_api_url(base_url, "videos")
        
        elif api_type == "openai_chat":
            return self._normalize_api_url(base_url, "chat")
        
        elif api_type == "openai_images":
            return self._normalize_api_url(base_url, "images")
        
        return base_url

    def _parse_stream_response(self, resp_text: str) -> Optional[Dict]:
        """解析流式响应（SSE 格式）"""
        if "data: " not in resp_text:
            return None
        
        full_content = ""
        tool_calls_buffer = {}
        valid_stream = False
        
        for line in resp_text.splitlines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    valid_stream = True
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        
                        if "content" in delta and delta["content"]:
                            full_content += delta["content"]
                        
                        if "tool_calls" in delta and delta["tool_calls"]:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_buffer:
                                    tool_calls_buffer[idx] = ""
                                if "function" in tc and "arguments" in tc["function"]:
                                    tool_calls_buffer[idx] += tc["function"]["arguments"]
                except:
                    pass
        
        if not valid_stream:
            return None
        
        msg_obj = {"content": full_content, "role": "assistant"}
        
        if tool_calls_buffer:
            msg_obj["tool_calls"] = []
            for idx in sorted(tool_calls_buffer.keys()):
                msg_obj["tool_calls"].append({
                    "function": {"arguments": tool_calls_buffer[idx]}
                })
        
        return {"choices": [{"message": msg_obj, "finish_reason": "stop"}]}

    # ================= 统一调用接口 =================

    async def generate_video(
        self,
        prompt: str,
        params: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        proxy: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        统一的视频生成接口
        
        根据配置自动选择 API 类型
        
        Args:
            prompt: 提示词
            params: 额外参数
            image_bytes: 图片数据
            proxy: 代理地址
            
        Returns:
            (video_url, message)
        """
        api_mode = self.config.get("api_mode", "plato")
        
        if api_mode == "openai":
            return await self.call_openai_video_api(prompt, params, image_bytes, proxy)
        else:
            # 柏拉图 API 需要先提交任务再轮询
            task_id, error = await self.submit_plato_task(prompt, params, image_bytes, proxy)
            if not task_id:
                return None, error
            return await self.poll_plato_result(task_id, proxy)
