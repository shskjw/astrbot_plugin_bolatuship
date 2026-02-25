"""
astrbot_plugin_bolatuship - Sora 视频生成插件

支持:
- 柏拉图 API (原生接口)
- OpenAI 兼容格式 API
- LLM 智能判断生成视频
- 文生视频 / 图生视频
"""

import asyncio
import base64
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import aiohttp
import aiofiles
import aiofiles.os
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core import AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.core.platform.astr_message_event import AstrMessageEvent

# 导入模块
from .api_manager import ApiManager
from .data_manager import DataManager, norm_id
from .context_manager import ContextManager, LLMTaskAnalyzer


@register(
    "astrbot_plugin_bolatuship",
    "shskjw",
    "Sora 视频生成插件，支持柏拉图API和OpenAI兼容格式，支持LLM智能判断",
    "2.0.0",
    "https://github.com/shkjw/astrbot_plugin_bolatuship",
)
class VideoGenPlugin(Star):
    """
    Sora 视频生成插件
    
    功能:
    - 文生视频: 根据文字描述生成视频
    - 图生视频: 将图片转换为视频
    - LLM 智能判断: 自动识别用户意图
    - 支持多种 API 格式
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        
        # 初始化管理器
        self.data_mgr = DataManager(self.plugin_data_dir, config)
        self.api_mgr = ApiManager(config)
        self.ctx_mgr = ContextManager(
            max_messages=config.get("context_max_messages", 50),
            max_sessions=config.get("context_max_sessions", 100)
        )
        
        # LLM 智能判断配置
        self._llm_auto_detect = config.get("enable_llm_auto_detect", False)
        self._context_rounds = config.get("context_rounds", 20)
        self._auto_detect_confidence = config.get("auto_detect_confidence", 0.8)
        
        # HTTP 客户端
        self._session: Optional[aiohttp.ClientSession] = None
        self._proxy: Optional[str] = None

    async def initialize(self):
        """初始化插件"""
        # 初始化数据管理器
        await self.data_mgr.initialize()
        
        # 设置代理
        if self.conf.get("use_proxy", False):
            self._proxy = self.conf.get("proxy_url")
            if self._proxy:
                logger.info(f"[VideoGen] 使用代理: {self._proxy}")
        
        # 检查 API 配置
        if not self.conf.get("api_keys"):
            logger.warning("[VideoGen] 未配置任何 API 密钥，插件无法工作")
        
        api_mode = self.conf.get("api_mode", "plato")
        logger.info(f"[VideoGen] 插件已加载 v2.0.0 | API模式: {api_mode} | LLM智能判断: {'已启用' if self._llm_auto_detect else '未启用'}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取 HTTP Session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def terminate(self):
        """终止插件"""
        await self.api_mgr.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("[VideoGen] 插件已终止")

    # ================= 工具函数 =================

    def is_admin(self, event: AstrMessageEvent) -> bool:
        """检查是否是管理员"""
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _get_bot_id(self, event: AstrMessageEvent) -> str:
        """获取机器人 ID"""
        if hasattr(event, "self_id") and event.self_id:
            return str(event.self_id)
        if hasattr(self.context, "get_self_id"):
            try:
                sid = self.context.get_self_id()
                if sid:
                    return str(sid)
            except:
                pass
        return ""

    async def _load_image_bytes(self, src: str) -> Optional[bytes]:
        """加载图片数据"""
        try:
            if Path(src).is_file():
                async with aiofiles.open(src, 'rb') as f:
                    return await f.read()
            elif src.startswith("http"):
                session = await self._get_session()
                async with session.get(src, proxy=self._proxy, timeout=120) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            elif src.startswith("base64://"):
                return base64.b64decode(src[9:])
        except Exception as e:
            logger.error(f"[VideoGen] 加载图片失败: {e}")
        return None

    async def _find_image_in_segments(self, segments: List[Any]) -> Optional[bytes]:
        """从消息段中查找图片"""
        for seg in segments:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_image_bytes(seg.url)):
                    return img
                if seg.file and (img := await self._load_image_bytes(seg.file)):
                    return img
        return None

    async def get_image_from_event(self, event: AstrMessageEvent) -> Optional[bytes]:
        """从事件中获取图片"""
        # 检查回复消息中的图片
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Reply) and seg.chain:
                if image_bytes := await self._find_image_in_segments(seg.chain):
                    return image_bytes
        # 检查当前消息中的图片
        return await self._find_image_in_segments(event.message_obj.message)

    def _parse_prompt_params(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """解析提示词中的参数"""
        params = {}
        clean_prompt = prompt

        # 匹配括号中的参数
        param_match = re.search(r'[\(（](.*?)[\)）]', prompt)
        if param_match:
            param_str = param_match.group(1)
            clean_prompt = re.sub(r'[\(（].*?[\)）]', '', prompt).strip()

            if '竖屏' in param_str or '9:16' in param_str:
                params['aspect_ratio'] = '9:16'
            elif '横屏' in param_str or '16:9' in param_str:
                params['aspect_ratio'] = '16:9'
            if '高清' in param_str or 'hd' in param_str.lower():
                params['hd'] = True
            duration_match = re.search(r'(\d+)[s秒]', param_str)
            if duration_match:
                params['duration'] = duration_match.group(1)

        return clean_prompt, params

    def _get_quota_str(self, deduction: dict, uid: str) -> str:
        """获取剩余次数字符串"""
        if deduction["source"] == "free":
            return "∞"
        return str(self.data_mgr.get_user_count(uid))

    async def _download_video(self, url: str) -> Optional[str]:
        """下载视频到本地"""
        filename = f"sora_video_{uuid.uuid4()}.mp4"
        filepath = str(self.plugin_data_dir / filename)
        try:
            session = await self._get_session()
            async with session.get(url, proxy=self._proxy, timeout=300) as resp:
                resp.raise_for_status()
                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        await f.write(chunk)
            return filepath
        except Exception as e:
            logger.error(f"[VideoGen] 下载视频失败: {e}")
            if await aiofiles.os.path.exists(filepath):
                await aiofiles.os.remove(filepath)
            return None

    # ================= 核心生成逻辑 =================

    async def _generate_video_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        hide_progress: bool = False
    ):
        """执行视频生成任务"""
        sender_id = norm_id(event.get_sender_id())
        group_id = norm_id(event.get_group_id()) if event.get_group_id() else None

        # 解析参数
        clean_prompt, params = self._parse_prompt_params(prompt)

        # 检查权限
        deduction = await self.data_mgr.check_quota(
            sender_id, group_id, self.is_admin(event), cost=1
        )
        if not deduction["allowed"]:
            yield event.plain_result(deduction["msg"])
            return

        # 发送进度提示
        mode_str = "图生视频" if image_bytes else "文生视频"
        if not hide_progress:
            yield event.plain_result(f"✅ 任务已提交 ({mode_str})，正在排队生成...")

        # 调用 API 生成视频
        video_url, status_msg = await self.api_mgr.generate_video(
            clean_prompt, params, image_bytes, self._proxy
        )

        if not video_url:
            yield event.plain_result(f"❌ 生成失败: {status_msg}")
            return

        if not hide_progress:
            yield event.plain_result("✅ 生成成功，正在下载视频...")

        # 下载视频
        filepath = await self._download_video(video_url)
        if not filepath:
            yield event.plain_result(f"❌ 视频下载失败，请尝试手动下载:\n{video_url}")
            return

        # 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(sender_id, 1)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(group_id, 1)

        # 记录使用
        await self.data_mgr.record_usage(sender_id, group_id)

        # 发送视频
        try:
            video_component = Comp.File(file=filepath, name="sora_video.mp4")
            quota_str = self._get_quota_str(deduction, sender_id)
            caption = f"🎬 视频已生成！\n剩余次数: {quota_str}"
            yield event.chain_result([video_component, Comp.Plain(caption)])
        except Exception as e:
            logger.error(f"[VideoGen] 发送视频失败: {e}", exc_info=True)
            yield event.plain_result(f"🎬 文件发送失败，请点击链接下载：\n{video_url}")
        finally:
            # 清理临时文件
            if await aiofiles.os.path.exists(filepath):
                await aiofiles.os.remove(filepath)

    # ================= LLM 工具 =================

    @filter.llm_tool(name="video_generate_text")
    async def text_to_video_tool(self, event: AstrMessageEvent, prompt: str):
        '''根据文本描述生成视频（文生视频）。
        
        调用前请判断用户是否明确要求生成视频。如果用户只是闲聊则不要调用。
        
        Args:
            prompt(string): 视频生成的提示词，描述你想要生成的视频内容。
        '''
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"

        sender_id = norm_id(event.get_sender_id())
        group_id = norm_id(event.get_group_id()) if event.get_group_id() else None

        # 检查权限
        deduction = await self.data_mgr.check_quota(
            sender_id, group_id, self.is_admin(event), cost=1
        )
        if not deduction["allowed"]:
            return deduction["msg"]

        # 发送进度提示
        if self.conf.get("llm_show_progress", True):
            await event.send(event.chain_result([Comp.Plain(f"🎬 收到文生视频请求，正在生成...")]))

        # 启动后台任务
        asyncio.create_task(self._run_llm_video_task(event, prompt, None, deduction, sender_id, group_id))

        return "[TOOL_SUCCESS] 文生视频任务已启动。视频将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可。"

    @filter.llm_tool(name="video_generate_image")
    async def image_to_video_tool(self, event: AstrMessageEvent, prompt: str = ""):
        '''将用户发送的图片转换为视频（图生视频）。仅在用户明确要求将图片转为视频时才调用。
        
        调用前请判断：
        1. 用户是否明确要求将图片转换为视频？
        2. 用户是否发送了图片或引用了包含图片的消息？
        
        Args:
            prompt(string): 视频生成的提示词（可选），描述视频的动作或效果。
        '''
        if not self.conf.get("enable_llm_tool", True):
            return "❌ LLM 工具已禁用，请使用指令模式调用此功能。"

        sender_id = norm_id(event.get_sender_id())
        group_id = norm_id(event.get_group_id()) if event.get_group_id() else None

        # 提取图片
        image_bytes = await self.get_image_from_event(event)
        if not image_bytes:
            return "❌ 未检测到图片，请发送或引用图片后再试。"

        # 检查权限
        deduction = await self.data_mgr.check_quota(
            sender_id, group_id, self.is_admin(event), cost=1
        )
        if not deduction["allowed"]:
            return deduction["msg"]

        # 发送进度提示
        if self.conf.get("llm_show_progress", True):
            await event.send(event.chain_result([Comp.Plain(f"🎬 收到图生视频请求，正在生成...")]))

        # 启动后台任务
        asyncio.create_task(self._run_llm_video_task(event, prompt or "让图片动起来", image_bytes, deduction, sender_id, group_id))

        return "[TOOL_SUCCESS] 图生视频任务已启动。视频将在后台生成并自动发送给用户。【重要】你不需要再回复任何内容，保持沉默即可。"

    async def _run_llm_video_task(
        self,
        event: AstrMessageEvent,
        prompt: str,
        image_bytes: Optional[bytes],
        deduction: dict,
        sender_id: str,
        group_id: Optional[str]
    ):
        """LLM 工具的后台视频生成任务"""
        try:
            # 解析参数
            clean_prompt, params = self._parse_prompt_params(prompt)

            # 调用 API
            video_url, status_msg = await self.api_mgr.generate_video(
                clean_prompt, params, image_bytes, self._proxy
            )

            if not video_url:
                await event.send(event.chain_result([Comp.Plain(f"❌ 生成失败: {status_msg}")]))
                return

            # 下载视频
            filepath = await self._download_video(video_url)
            if not filepath:
                await event.send(event.chain_result([Comp.Plain(f"❌ 视频下载失败，请手动下载:\n{video_url}")]))
                return

            # 扣费
            if deduction["source"] == "user":
                await self.data_mgr.decrease_user_count(sender_id, 1)
            elif deduction["source"] == "group":
                await self.data_mgr.decrease_group_count(group_id, 1)

            await self.data_mgr.record_usage(sender_id, group_id)

            # 发送视频
            try:
                video_component = Comp.File(file=filepath, name="sora_video.mp4")
                quota_str = self._get_quota_str(deduction, sender_id)
                caption = f"🎬 视频已生成！\n剩余次数: {quota_str}"
                await event.send(event.chain_result([video_component, Comp.Plain(caption)]))
            except Exception as e:
                logger.error(f"[VideoGen] 发送视频失败: {e}")
                await event.send(event.chain_result([Comp.Plain(f"🎬 文件发送失败，请下载:\n{video_url}")]))
            finally:
                if await aiofiles.os.path.exists(filepath):
                    await aiofiles.os.remove(filepath)

        except Exception as e:
            logger.error(f"[VideoGen] LLM 任务错误: {e}", exc_info=True)
            await event.send(event.chain_result([Comp.Plain(f"❌ 系统错误: {e}")]))

    # ================= 传统指令 =================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_video_request(self, event: AstrMessageEvent):
        """处理视频生成请求"""
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text:
            return

        parts = text.split()
        cmd = parts[0].strip()
        custom_prefix = self.conf.get("extra_prefix", "生成视频")

        prompt = ""
        if cmd == custom_prefix:
            prompt = text.removeprefix(cmd).strip()
            if not prompt:
                return
        elif cmd in self.data_mgr.prompt_map:
            prompt = self.data_mgr.prompt_map[cmd]
            additional = text.removeprefix(cmd).strip()
            if additional:
                prompt = f"{prompt}, {additional}"
        else:
            return

        # 获取图片
        image_bytes = await self.get_image_from_event(event)

        async for result in self._generate_video_task(event, prompt, image_bytes):
            yield result

        event.stop_event()

    @filter.command("视频签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        """签到获取次数"""
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 本机器人未开启签到功能。")
            return
        
        uid = norm_id(event.get_sender_id())
        msg = await self.data_mgr.process_checkin(uid)
        yield event.plain_result(msg)

    @filter.command("视频查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        """查询剩余次数"""
        uid = norm_id(event.get_sender_id())
        user_count = self.data_mgr.get_user_count(uid)
        reply = f"您当前个人剩余次数: {user_count}"
        
        gid = event.get_group_id()
        if gid and self.conf.get("enable_group_limit", False):
            group_count = self.data_mgr.get_group_count(norm_id(gid))
            reply += f"\n本群共享剩余次数: {group_count}"
        
        yield event.plain_result(reply)

    @filter.command("视频帮助", prefix_optional=True)
    async def on_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        custom_prefix = self.conf.get("extra_prefix", "生成视频")
        presets = list(self.data_mgr.prompt_map.keys())
        
        help_text = (
            f"🎬 Sora 视频生成插件帮助 (v2.0.0)\n\n"
            f"【使用方法】\n"
            f"1. 发送图片或引用图片，然后输入指令 (图生视频)\n"
            f"2. 不带图片直接使用指令 (文生视频)\n"
            f"3. 在提示词末尾用括号添加参数：\n"
            f"   `(竖屏, 15s, 高清)`\n\n"
            f"【指令】\n"
            f"自定义: #{custom_prefix} <描述>\n"
        )
        
        if presets:
            help_text += f"预设: #{'、#'.join(presets[:5])}\n"
        
        help_text += (
            f"\n【其他】\n"
            f"#视频签到 - 获取免费次数\n"
            f"#视频查询次数 - 查看剩余次数\n"
        )
        
        yield event.plain_result(help_text)

    @filter.command("视频预设列表", prefix_optional=True)
    async def on_preset_list(self, event: AstrMessageEvent):
        """显示预设列表"""
        if not self.data_mgr.prompt_map:
            yield event.plain_result("暂无预设。")
            return
        
        msg = "📋 当前预设列表:\n"
        for key in self.data_mgr.prompt_map.keys():
            msg += f"- {key}\n"
        yield event.plain_result(msg)

    # ================= 管理员指令 =================

    @filter.command("视频添加预设", prefix_optional=True)
    async def on_add_preset(self, event: AstrMessageEvent):
        """添加预设"""
        if not self.is_admin(event):
            return
        
        raw = event.message_str.strip().removeprefix("视频添加预设").strip()
        if ":" not in raw:
            yield event.plain_result('格式错误，示例:\n#视频添加预设 电影感:cinematic, epic, 4k')
            return
        
        key, value = map(str.strip, raw.split(":", 1))
        await self.data_mgr.add_user_prompt(key, value)
        
        # 同步到配置
        prompt_list = self.conf.get("prompt_list", [])
        prompt_list = [item for item in prompt_list if not item.startswith(f"{key}:")]
        prompt_list.append(f"{key}:{value}")
        self.conf["prompt_list"] = prompt_list
        
        yield event.plain_result(f"✅ 已保存预设: {key}")

    @filter.command("视频删除预设", prefix_optional=True)
    async def on_delete_preset(self, event: AstrMessageEvent):
        """删除预设"""
        if not self.is_admin(event):
            return
        
        key = event.message_str.strip().removeprefix("视频删除预设").strip()
        if await self.data_mgr.delete_user_prompt(key):
            # 同步到配置
            prompt_list = self.conf.get("prompt_list", [])
            prompt_list = [item for item in prompt_list if not item.startswith(f"{key}:")]
            self.conf["prompt_list"] = prompt_list
            yield event.plain_result(f"✅ 已删除预设: {key}")
        else:
            yield event.plain_result(f"❌ 未找到预设: {key}")

    @filter.command("视频增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        """增加用户次数"""
        if not self.is_admin(event):
            return
        
        args = event.message_str.strip().removeprefix("视频增加用户次数").strip()
        match = re.fullmatch(r"(\d+)\s+(\d+)", args)
        if not match:
            yield event.plain_result('格式错误: #视频增加用户次数 <QQ号> <次数>')
            return
        
        target_qq, count = match.group(1), int(match.group(2))
        await self.data_mgr.add_user_count(target_qq, count)
        new_count = self.data_mgr.get_user_count(target_qq)
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，当前剩余 {new_count} 次。")

    @filter.command("视频增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        """增加群组次数"""
        if not self.is_admin(event):
            return
        
        args = event.message_str.strip().removeprefix("视频增加群组次数").strip()
        match = re.fullmatch(r"(\d+)\s+(\d+)", args)
        if not match:
            yield event.plain_result('格式错误: #视频增加群组次数 <群号> <次数>')
            return
        
        target_group, count = match.group(1), int(match.group(2))
        await self.data_mgr.add_group_count(target_group, count)
        new_count = self.data_mgr.get_group_count(target_group)
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，当前剩余 {new_count} 次。")

    @filter.command("视频清除缓存", prefix_optional=True)
    async def on_clear_cache(self, event: AstrMessageEvent):
        """清除缓存"""
        if not self.is_admin(event):
            return

        count = 0
        try:
            if not await aiofiles.os.path.isdir(self.plugin_data_dir):
                yield event.plain_result("ℹ️ 缓存目录不存在。")
                return

            for filename in await aiofiles.os.listdir(self.plugin_data_dir):
                if filename.startswith("sora_video_") and filename.endswith(".mp4"):
                    filepath = self.plugin_data_dir / filename
                    await aiofiles.os.remove(filepath)
                    count += 1

            yield event.plain_result(f"✅ 已清除 {count} 个临时视频文件。")
        except Exception as e:
            logger.error(f"[VideoGen] 清除缓存错误: {e}", exc_info=True)
            yield event.plain_result(f"❌ 清除缓存时发生错误。")

    @filter.command("视频今日统计", prefix_optional=True)
    async def on_daily_stats(self, event: AstrMessageEvent):
        """查看今日统计"""
        if not self.is_admin(event):
            return
        
        stats = self.data_mgr.get_daily_stats()
        today = stats.get("date", "无数据")
        
        if not stats.get("users"):
            yield event.plain_result(f"📊 {today} 暂无使用记录")
            return
        
        u_top = sorted(stats["users"].items(), key=lambda x: x[1], reverse=True)[:10]
        g_top = sorted(stats.get("groups", {}).items(), key=lambda x: x[1], reverse=True)[:10]
        
        msg = f"📊 {today} 统计:\n"
        msg += "👥 群排行:\n" + ("\n".join([f"  {k}: {v}" for k, v in g_top]) or "  无")
        msg += "\n\n👤 用户排行:\n" + ("\n".join([f"  {k}: {v}" for k, v in u_top]) or "  无")
        
        yield event.plain_result(msg)

    # ================= 上下文记录 =================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message_record(self, event: AstrMessageEvent):
        """记录消息到上下文（高优先级，不阻断）"""
        try:
            session_id = event.unified_msg_origin
            msg_id = str(event.message_obj.message_id)
            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name() or sender_id
            
            bot_id = self._get_bot_id(event)
            is_bot = (sender_id == bot_id) if bot_id else False
            
            # 检查消息中是否有图片
            has_image = False
            image_urls = []
            for seg in event.message_obj.message:
                if isinstance(seg, Comp.Image):
                    has_image = True
                    if seg.url:
                        image_urls.append(seg.url)
                    elif seg.file:
                        image_urls.append(seg.file)
            
            # 记录到上下文管理器
            await self.ctx_mgr.add_message(
                session_id=session_id,
                msg_id=msg_id,
                sender_id=sender_id,
                sender_name=sender_name,
                content=event.message_str[:500],
                is_bot=is_bot,
                has_image=has_image,
                image_urls=image_urls
            )
        except Exception as e:
            logger.debug(f"[VideoGen] 消息记录失败: {e}")
        
        # 不阻断事件传递
        return
