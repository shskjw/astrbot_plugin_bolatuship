"""
上下文管理器 - 管理聊天记录和 LLM 智能判断

用于:
- 记录聊天历史
- LLM 智能判断是否需要生成视频
- 分析用户意图
"""

import asyncio
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from astrbot.api import logger


@dataclass(slots=True)
class MessageRecord:
    """消息记录"""
    msg_id: str
    sender_id: str
    sender_name: str
    content: str
    timestamp: float
    is_bot: bool = False
    has_image: bool = False
    image_urls: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SessionState:
    """会话状态"""
    messages: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = 0.0


class ContextManager:
    """
    上下文管理器 - 管理聊天记录
    
    功能:
    - 记录群聊/私聊消息历史
    - 提供上下文获取接口
    - 支持 LRU 淘汰机制
    """

    __slots__ = ("_sessions", "_locks", "_max_messages", "_max_sessions", "_cache_lock", "_bot_id")

    def __init__(self, max_messages: int = 50, max_sessions: int = 100) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._cache_lock = asyncio.Lock()
        self._max_messages = max(10, max_messages)
        self._max_sessions = max(10, max_sessions)
        self._bot_id: Optional[str] = None

    def set_bot_id(self, bot_id: str) -> None:
        """设置 Bot ID"""
        self._bot_id = bot_id

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """获取会话锁"""
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def _get_or_create_session(self, session_id: str) -> SessionState:
        """获取或创建会话状态"""
        async with self._cache_lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]

            # LRU 淘汰
            while len(self._sessions) >= self._max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                self._locks.pop(evicted_id, None)

            state = SessionState()
            state.messages = deque(maxlen=self._max_messages)
            self._sessions[session_id] = state
            return state

    async def add_message(
        self,
        session_id: str,
        msg_id: str,
        sender_id: str,
        sender_name: str,
        content: str,
        is_bot: bool = False,
        has_image: bool = False,
        image_urls: Optional[List[str]] = None
    ) -> None:
        """添加消息到会话"""
        async with self._get_lock(session_id):
            state = await self._get_or_create_session(session_id)
            
            msg = MessageRecord(
                msg_id=msg_id,
                sender_id=sender_id,
                sender_name=sender_name,
                content=content[:500],  # 限制长度
                timestamp=time.time(),
                is_bot=is_bot,
                has_image=has_image,
                image_urls=image_urls or []
            )
            
            state.messages.append(msg)
            state.last_updated = time.time()

    async def get_recent_messages(
        self,
        session_id: str,
        count: int = 20,
        include_bot: bool = True
    ) -> List[MessageRecord]:
        """获取最近的消息记录"""
        async with self._get_lock(session_id):
            if session_id not in self._sessions:
                return []
            
            state = self._sessions[session_id]
            messages = list(state.messages)
            
            if not include_bot:
                messages = [m for m in messages if not m.is_bot]
            
            return messages[-count:] if count > 0 else messages

    def get_formatted_context(
        self,
        messages: List[MessageRecord],
        max_chars: int = 3000
    ) -> str:
        """格式化消息为上下文字符串"""
        if not messages:
            return ""
        
        lines: List[str] = []
        total_chars = 0
        
        for msg in reversed(messages):
            sender = "[Bot]" if msg.is_bot else msg.sender_name
            image_tag = " [含图片]" if msg.has_image else ""
            line = f"{sender}: {msg.content}{image_tag}"
            
            if total_chars + len(line) > max_chars:
                break
            
            lines.insert(0, line)
            total_chars += len(line) + 1
        
        return "\n".join(lines)

    def has_recent_images(self, messages: List[MessageRecord], within_count: int = 5) -> bool:
        """检查最近消息中是否有图片"""
        recent = messages[-within_count:] if len(messages) > within_count else messages
        return any(m.has_image for m in recent)

    def get_last_image_message(self, messages: List[MessageRecord]) -> Optional[MessageRecord]:
        """获取最后一条包含图片的消息"""
        for msg in reversed(messages):
            if msg.has_image:
                return msg
        return None

    async def clear_session(self, session_id: str) -> int:
        """清除会话"""
        async with self._cache_lock:
            state = self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
            if not state:
                return 0
            return len(state.messages)

    def get_session_count(self) -> int:
        """获取当前会话数量"""
        return len(self._sessions)


class LLMTaskAnalyzer:
    """
    LLM 任务分析器 - 分析上下文决定任务类型
    
    任务类型:
    - text_to_video: 文生视频（无图片输入，纯文本描述）
    - image_to_video: 图生视频（有图片输入，需要处理图片）
    - none: 不需要生成视频
    
    注意：此分析器仅用于辅助判断，LLM工具的调用由LLM自己决定
    """

    # 强文生视频关键词（必须明确表达生成视频意图）
    STRONG_TEXT_TO_VIDEO_KEYWORDS = [
        "生成视频", "做个视频", "做一个视频", "创建视频", "制作视频",
        "generate video", "create video", "make video",
        "帮我生成视频", "给我生成视频", "来一个视频", "来个视频",
        "视频生成", "生成一个视频", "生成一段视频",
    ]
    
    # 弱文生视频关键词（需要结合上下文判断）
    WEAK_TEXT_TO_VIDEO_KEYWORDS = [
        "视频", "动画", "动态", "动图",
        "video", "animation", "animate",
    ]

    # 强图生视频关键词（明确表达图片转视频意图）
    STRONG_IMAGE_TO_VIDEO_KEYWORDS = [
        "图片生成视频", "图生视频", "把图片变成视频", "图片转视频",
        "把这张图做成视频", "把这个图做成视频", "这张图生成视频",
        "image to video", "img2video", "i2v",
        "让图片动起来", "让这张图动起来", "图片动画化",
    ]
    
    # 弱图生视频关键词（需要有图片才考虑）
    WEAK_IMAGE_TO_VIDEO_KEYWORDS = [
        "动起来", "变成视频", "转成视频", "做成视频",
        "animate", "to video",
    ]

    # 明确不需要生成视频的关键词
    NO_VIDEO_KEYWORDS = [
        "查询", "帮助", "列表", "统计", "签到", "次数",
        "怎么用", "什么是", "为什么", "如何使用", "教程",
        "help", "list", "query", "how to", "what is", "why",
        "你好", "谢谢", "好的", "嗯", "哦", "啊",
        "hello", "hi", "thanks", "ok", "yes", "no",
    ]
    
    # 闲聊关键词（明确是闲聊，不需要生成视频）
    CHAT_KEYWORDS = [
        "你是谁", "你叫什么", "你在干嘛", "你好吗", "吃了吗",
        "天气", "时间", "日期", "新闻", "今天",
        "哈哈", "笑死", "真的吗", "是吗", "好吧",
    ]

    @classmethod
    def analyze_task_type(
        cls,
        current_message: str,
        context_messages: List[MessageRecord],
        has_current_image: bool = False
    ) -> Dict[str, Any]:
        """
        分析任务类型（严格模式）
        
        只有在非常明确的情况下才返回需要生成视频的结果
        
        Returns:
            {
                "task_type": "text_to_video" | "image_to_video" | "none",
                "confidence": float (0-1),
                "reason": str,
                "suggested_prompt": str (可选)
            }
        """
        result = {
            "task_type": "none",
            "confidence": 0.0,
            "reason": "默认不生成视频",
            "suggested_prompt": ""
        }
        
        msg_lower = current_message.lower().strip()
        
        # 消息太短，可能是闲聊
        if len(msg_lower) < 3:
            result["reason"] = "消息太短，判断为闲聊"
            return result
        
        # 1. 检查是否是闲聊
        for kw in cls.CHAT_KEYWORDS:
            if kw in msg_lower:
                result["reason"] = f"检测到闲聊关键词: {kw}"
                return result
        
        # 2. 检查是否明确不需要生成视频
        for kw in cls.NO_VIDEO_KEYWORDS:
            if kw in msg_lower:
                result["reason"] = f"检测到非生成视频关键词: {kw}"
                return result
        
        # 3. 检查是否有图片输入
        has_context_image = any(m.has_image for m in context_messages[-3:]) if context_messages else False
        has_any_image = has_current_image or has_context_image
        
        # 4. 检查强图生视频关键词（优先级最高）
        for kw in cls.STRONG_IMAGE_TO_VIDEO_KEYWORDS:
            if kw in msg_lower:
                result["task_type"] = "image_to_video"
                result["confidence"] = 0.9
                result["reason"] = f"检测到强图生视频关键词: {kw}"
                result["suggested_prompt"] = current_message
                return result
        
        # 5. 检查强文生视频关键词
        for kw in cls.STRONG_TEXT_TO_VIDEO_KEYWORDS:
            if kw in msg_lower:
                result["task_type"] = "text_to_video"
                result["confidence"] = 0.85
                result["reason"] = f"检测到强文生视频关键词: {kw}"
                result["suggested_prompt"] = current_message
                return result
        
        # 6. 有图片 + 弱图生视频关键词
        if has_any_image:
            weak_i2v_count = sum(1 for kw in cls.WEAK_IMAGE_TO_VIDEO_KEYWORDS if kw in msg_lower)
            if weak_i2v_count >= 1:
                result["task_type"] = "image_to_video"
                result["confidence"] = min(0.7, 0.5 + weak_i2v_count * 0.1)
                result["reason"] = f"有图片且检测到弱图生视频关键词"
                result["suggested_prompt"] = current_message
                return result
        
        # 7. 无图片 + 弱文生视频关键词（需要多个关键词才触发）
        if not has_any_image:
            weak_t2v_count = sum(1 for kw in cls.WEAK_TEXT_TO_VIDEO_KEYWORDS if kw in msg_lower)
            if weak_t2v_count >= 2:  # 需要至少2个弱关键词
                result["task_type"] = "text_to_video"
                result["confidence"] = min(0.6, 0.3 + weak_t2v_count * 0.1)
                result["reason"] = f"检测到{weak_t2v_count}个弱文生视频关键词"
                result["suggested_prompt"] = current_message
                return result
        
        # 8. 默认不生成视频
        if has_any_image:
            result["reason"] = "有图片但无明确处理意图，不自动触发"
        else:
            result["reason"] = "无明确生成视频意图"
        
        return result

    @classmethod
    def build_analysis_prompt(
        cls,
        current_message: str,
        context: str,
        has_image: bool
    ) -> str:
        """
        构建用于 LLM 分析的提示词
        
        用于让 LLM 自己判断是否需要生成视频
        """
        image_status = "用户消息中包含图片" if has_image else "用户消息中没有图片"
        
        prompt = f"""请分析以下对话，判断用户是否需要生成视频。

## 上下文对话记录:
{context}

## 当前用户消息:
{current_message}

## 图片状态:
{image_status}

## 请判断:
1. 用户是否需要生成视频？(是/否)
2. 如果需要，是哪种类型？
   - text_to_video: 文生视频（根据文字描述生成视频）
   - image_to_video: 图生视频（将图片转换为视频）
3. 置信度 (0-100)
4. 建议的提示词（如果需要生成视频）

请以 JSON 格式回复:
{{
    "need_video": true/false,
    "task_type": "text_to_video" | "image_to_video" | "none",
    "confidence": 0-100,
    "reason": "判断理由",
    "suggested_prompt": "建议的提示词"
}}"""
        
        return prompt
