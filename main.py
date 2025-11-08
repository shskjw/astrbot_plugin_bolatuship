# --- main.py (v1.0.6 - 移除自动模型切换) ---

import asyncio
import base64
import json
import time
import os
import uuid
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import io

try:
    from PIL import Image
except ImportError:
    Image = None

import aiohttp
import aiofiles
import aiofiles.os
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.core.platform.astr_message_event import AstrMessageEvent


class PlatoSoraPlugin(Star):
    """
    astrbot_plugin_plato_sora by shskjw
    Version: 1.0.0 (Removed auto model switching, default 15s)
    Description: 使用柏拉图API接口和Sora 2等模型生成视频。
    """


    class APIClient:
        def __init__(self, proxy_url: Optional[str] = None):
            self.proxy = proxy_url
            self.session = aiohttp.ClientSession()
            if self.proxy:
                logger.info(f"[PlatoSora] APIClient 使用代理: {self.proxy}")

        async def _download_media(self, url: str) -> Optional[bytes]:
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=120) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as e:
                logger.error(f"[PlatoSora] 媒体下载失败: {e}", exc_info=True)
                return None

        async def _load_bytes(self, src: str) -> Optional[bytes]:
            if Path(src).is_file():
                try:
                    async with aiofiles.open(src, 'rb') as f:
                        return await f.read()
                except Exception as e:
                    logger.error(f"读取本地图片文件失败: {src}, error: {e}")
                    return None
            elif src.startswith("http"):
                return await self._download_media(src)
            elif src.startswith("base64://"):
                return base64.b64decode(src[9:])
            return None

        async def _find_image_in_segments(self, segments: List[Any]) -> Optional[bytes]:
            for seg in segments:
                if isinstance(seg, Comp.Image):
                    if seg.url and (img := await self._load_bytes(seg.url)): return img
                    if seg.file and (img := await self._load_bytes(seg.file)): return img
            return None

        async def get_image_from_event(self, event: AstrMessageEvent) -> Optional[bytes]:
            for seg in event.message_obj.message:
                if isinstance(seg, Comp.Reply) and seg.chain:
                    if image_bytes := await self._find_image_in_segments(seg.chain):
                        return image_bytes
            return await self._find_image_in_segments(event.message_obj.message)

        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "sora_user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "sora_group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_file = self.plugin_data_dir / "sora_user_checkin.json"
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.count_lock = asyncio.Lock()
        self.api_client: Optional[PlatoSoraPlugin.APIClient] = None

    async def initialize(self):
        if Image is None:
            logger.error("[PlatoSora] 图像处理库 Pillow 未安装，无法使用图片比例自动识别功能。")
            logger.error("[PlatoSora] 请在您的环境中使用 `pip install Pillow` 命令安装。")
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.api_client = self.APIClient(proxy_url)
        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()
        logger.info("Plato Sora 2 视频插件已加载 (v3.0.6 - no model switch)")
        if not self.conf.get("api_keys"):
            logger.warning("[PlatoSora] 未配置任何 API 密钥，插件无法工作")

    async def _load_prompt_map(self):
        self.prompt_map.clear()
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            if ":" in item:
                key, value = item.split(":", 1)
                self.prompt_map[key.strip()] = value.strip()
        logger.info(f"[PlatoSora] 加载了 {len(self.prompt_map)} 个指令预设。")

    # --- 数据读写 ---
    async def _load_data(self, file_path: Path) -> Dict:
        if not await aiofiles.os.path.exists(file_path): return {}
        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
            return {str(k): v for k, v in json.loads(content).items()}
        except Exception as e:
            logger.error(f"加载JSON文件 {file_path.name} 失败: {e}");
            return {}

    async def _save_data(self, file_path: Path, data: Dict):
        try:
            async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=4))
        except Exception as e:
            logger.error(f"保存JSON文件 {file_path.name} 失败: {e}")

    async def _load_user_counts(self):
        self.user_counts = await self._load_data(self.user_counts_file)

    async def _save_user_counts(self):
        await self._save_data(self.user_counts_file, self.user_counts)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(user_id, 0)

    async def _load_group_counts(self):
        self.group_counts = await self._load_data(self.group_counts_file)

    async def _save_group_counts(self):
        await self._save_data(self.group_counts_file, self.group_counts)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(group_id, 0)

    async def _load_user_checkin_data(self):
        self.user_checkin_data = await self._load_data(self.user_checkin_file)

    async def _save_user_checkin_data(self):
        await self._save_data(self.user_checkin_file, self.user_checkin_data)

    async def _decrease_user_count(self, user_id: str):
        async with self.count_lock:
            count = self._get_user_count(user_id)
            if count > 0: self.user_counts[user_id] = count - 1; await self._save_user_counts()

    async def _decrease_group_count(self, group_id: str):
        async with self.count_lock:
            count = self._get_group_count(group_id)
            if count > 0: self.group_counts[group_id] = count - 1; await self._save_group_counts()

    # --- 核心指令处理器 ---
    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_video_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command: return

        text = event.message_str.strip()
        if not text: return

        parts = text.split()
        cmd = parts[0].strip()
        custom_prompt_prefix = self.conf.get("extra_prefix", "生成视频")

        prompt = ""
        if cmd == custom_prompt_prefix:
            prompt = text.removeprefix(cmd).strip()
            if not prompt: return
        elif cmd in self.prompt_map:
            prompt = self.prompt_map[cmd]
            additional_prompt = text.removeprefix(cmd).strip()
            if additional_prompt: prompt = f"{prompt}, {additional_prompt}"
        else:
            return

        can_proceed, error_message = await self._check_permissions(event)
        if not can_proceed:
            if error_message: yield event.plain_result(error_message)
            return

        async for result in self._generate_video_task(event, prompt):
            yield result

        event.stop_event()

    # --- 参数解析器 ---
    def _parse_prompt_for_params(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        params = {}
        clean_prompt = prompt

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
                params['duration'] = int(duration_match.group(1))

        return clean_prompt, params
    
    # --- 自动识别图片比例的函数 ---
    def _get_aspect_ratio_from_image(self, image_bytes: bytes) -> Optional[str]:
        if not Image:
            return None
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                if width == 0 or height == 0: return None
                ratio = width / height
                if ratio > 1.1: return "16:9"
                elif ratio < 0.9: return "9:16"
                else: return "1:1"
        except Exception as e:
            logger.warning(f"[PlatoSora] 使用 Pillow 自动识别图片比例失败: {e}")
            return None

    # --- 核心生成与发送逻辑 ---
    async def _generate_video_task(self, event: AstrMessageEvent, prompt: str):
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()

        image_bytes = await self.api_client.get_image_from_event(event)
        clean_prompt, params = self._parse_prompt_for_params(prompt)

        if image_bytes and 'aspect_ratio' not in params:
            auto_ratio = self._get_aspect_ratio_from_image(image_bytes)
            if auto_ratio:
                params['aspect_ratio'] = auto_ratio
                logger.info(f"[PlatoSora] 自动识别图片比例为: {auto_ratio}")

        if 'duration' not in params:
            params['duration'] = 15
            logger.info("[PlatoSora] 未指定时长，使用默认值 15s。")


        yield event.plain_result(
            f"✅ 任务已提交 ({'图生视频' if image_bytes else '文生视频'})，正在排队生成...")

        task_id, error_msg = await self._submit_task(clean_prompt, params, image_bytes)
        if not task_id: yield event.plain_result(f"❌ 提交失败: {error_msg}"); return

        yield event.plain_result(f"⏳ 任务ID: {task_id}，正在生成中，请耐心等待...")

        video_url, status_msg = await self._poll_for_result(task_id)
        if not video_url: yield event.plain_result(f"❌ 处理失败: {status_msg}"); return

        yield event.plain_result("✅ 生成成功，正在下载视频到本地...")
        filepath = await self._download_video_async(video_url)
        if not filepath: yield event.plain_result(f"❌ 视频下载失败，请尝试手动下载:\n{video_url}"); return

        yield event.plain_result("✅ 下载完成，正在发送文件...")

        if not self.is_global_admin(event):
            if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                await self._decrease_group_count(group_id)
            elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                await self._decrease_user_count(sender_id)

        try:
            video_component = Comp.Video(file=f"file://{filepath}")
            caption_parts = []
            if self.is_global_admin(event):
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(
                    f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(
                    f"本群剩余: {self._get_group_count(group_id)}")
            caption_text = "🎬 Sora 视频已生成！"
            if caption_parts: caption_text += "\n\n" + " | ".join(caption_parts)
            yield event.chain_result([video_component, Comp.Plain(caption_text)])
        except Exception as e:
            logger.error(f"发送文件时失败: {e}", exc_info=True)
            yield event.plain_result(f"🎬 文件发送失败，请点击链接下载：\n{video_url}")
        finally:
            if await aiofiles.os.path.exists(filepath):
                await aiofiles.os.remove(filepath)
                logger.info(f"已清理临时文件: {filepath}")

    # --- API 交互 (v2接口) ---
    async def _submit_task(self, prompt: str, params: Dict[str, Any], image_bytes: Optional[bytes]) -> Tuple[
        Optional[str], str]:
        api_url = self.conf.get("api_url", "https://api.bltcy.ai")
        api_key = await self._get_api_key()
        if not api_key: return None, "无可用的 API Key"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # 【【【核心修改处】】】
        # 始终使用配置文件中指定的默认模型，不再自动切换
        model = self.conf.get("default_model", "sora-2")

        payload = {"prompt": prompt, "model": model, **params}
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            payload["images"] = [f"data:image/jpeg;base64,{base64_image}"]
        
        try:
            endpoint = f"{api_url}/v2/videos/generations"
            async with self.api_client.session.post(endpoint, json=payload, headers=headers,
                                                    proxy=self.api_client.proxy, timeout=180) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_info = data.get('error', {}).get('message', str(data))
                    return None, f"任务提交失败 ({resp.status}): {error_info}"
                task_id = data.get("task_id")
                if not task_id: return None, f"未能从响应中获取 task_id: {json.dumps(data)}"
                return task_id, "提交成功"
        except asyncio.TimeoutError:
            return None, "请求API超时，服务器可能正忙，请稍后再试"
        except Exception as e:
            logger.error(f"[PlatoSora] 任务提交网络错误: {e}", exc_info=True)
            return None, f"网络错误: {e}"

    # --- 状态轮询 (v2接口) ---
    async def _poll_for_result(self, task_id: str) -> Tuple[Optional[str], str]:
        api_key = await self._get_api_key()
        if not api_key: return None, "无可用的 API Key"
        api_url = self.conf.get("api_url", "https://api.bltcy.ai")
        timeout = self.conf.get("polling_timeout", 300)
        interval = self.conf.get("polling_interval", 5)
        start_time = time.monotonic()
        headers = {"Authorization": f"Bearer {api_key}"}
        endpoint = f"{api_url}/v2/videos/generations/{task_id}"
        while time.monotonic() - start_time < timeout:
            try:
                async with self.api_client.session.get(endpoint, headers=headers, proxy=self.api_client.proxy,
                                                       timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status = data.get("status")
                        if status == "SUCCESS":
                            video_url = data.get("data", {}).get("output")
                            if video_url: return video_url, "生成成功"
                            else: logger.error(f"成功但未找到视频链接: {json.dumps(data)}"); return None, "任务成功但响应中未找到视频链接"
                        elif status == "FAILURE":
                            reason = data.get("fail_reason", "未知错误")
                            try: reason = json.loads(reason).get("message", reason)
                            except: pass
                            return None, f"任务失败: {reason}"
                    else:
                        logger.warning(f"轮询状态码异常: {resp.status}, 响应: {await resp.text()}")
            except Exception as e:
                logger.warning(f"轮询状态时发生网络异常: {e}")
            await asyncio.sleep(interval)
        return None, "任务超时"

    # --- 其他指令 ---
    @filter.command("视频签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False): yield event.plain_result("📅 本机器人未开启签到功能。"); return
        user_id = event.get_sender_id()
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self.user_checkin_data.get(user_id) == today_str:
            yield event.plain_result(f"您今天已经签到过了！\n剩余次数: {self._get_user_count(user_id)}"); return
        reward = int(self.conf.get("checkin_fixed_reward", 3))
        if str(self.conf.get("enable_random_checkin", False)).lower() == 'true':
            max_reward = max(1, int(self.conf.get("checkin_random_reward_max", 5)))
            reward = random.randint(1, max_reward)
        current_count = self._get_user_count(user_id)
        new_count = current_count + reward
        self.user_counts[user_id] = new_count
        await self._save_user_counts()
        self.user_checkin_data[user_id] = today_str
        await self._save_user_checkin_data()
        yield event.plain_result(f"🎉 签到成功！获得 {reward} 次，当前剩余: {new_count} 次。")

    @filter.command("视频查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        user_count = self._get_user_count(user_id)
        reply_msg = f"您好，您当前个人剩余次数为: {user_count}"
        group_id = event.get_group_id()
        if group_id and self.conf.get("enable_group_limit", False):
            group_count = self._get_group_count(group_id)
            reply_msg += f"\n本群共享剩余次数为: {group_count}"
        yield event.plain_result(reply_msg)

    @filter.command("视频帮助", prefix_optional=True)
    async def on_cmd_help(self, event: AstrMessageEvent):
        custom_prefix = self.conf.get("extra_prefix", "生成视频")
        help_text = (f"【使用方法】\n"
                     f"1. 发送图片或引用图片，然后输入指令 (图生视频)。\n"
                     f"   (插件会自动识别比例, 默认生成15秒视频)\n"
                     f"2. 不带图片直接使用指令 (文生视频)。\n"
                     f"3. 在提示词末尾用括号添加高级参数，如：\n"
                     f"   `#(竖屏, 10s, 高清)` (可自定义时长等)\n\n"
                     f"【指令列表】\n"
                     f"自定义提示词: #{custom_prefix} <你的描述>\n"
                     f"预设指令: #{'、#'.join(self.prompt_map.keys())}\n\n"
                     f"【每日福利】\n#视频签到 - 获取免费次数\n\n"
                     f"【查询】\n#视频查询次数 - 查看剩余次数\n\n"
                     f"【管理】\n#视频清除缓存 (仅管理员)")
        yield event.plain_result(help_text)

    # --- 管理员指令 ---
    @filter.command("视频预设列表", prefix_optional=True)
    async def on_prompt_list(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        if not self.prompt_map: yield event.plain_result("暂无任何预设。"); return
        msg = "📋 当前预设指令列表:\n" + "\n".join(f"- {key}" for key in self.prompt_map.keys())
        yield event.plain_result(msg)

    @filter.command("视频添加预设", prefix_optional=True)
    async def on_add_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        raw = event.message_str.strip().removeprefix("视频添加预设").strip()
        if ":" not in raw:
            yield event.plain_result('格式错误, 示例:\n#视频添加预设 电影感:cinematic, epic, 4k')
            return
        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])
        found = False
        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"; found = True; break
        if not found: prompt_list.append(f"{key}:{new_value}")
        await self.conf.set("prompt_list", prompt_list)
        await self._load_prompt_map()
        yield event.plain_result(f"✅ 已保存预设:\n{key}:{new_value}")

    @filter.command("视频删除预设", prefix_optional=True)
    async def on_delete_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        key_to_delete = event.message_str.strip().removeprefix("视频删除预设").strip()
        prompt_list = self.conf.get("prompt_list", [])
        original_len = len(prompt_list)
        new_prompt_list = [item for item in prompt_list if not item.strip().startswith(key_to_delete + ":")]
        if len(new_prompt_list) < original_len:
            await self.conf.set("prompt_list", new_prompt_list)
            await self._load_prompt_map()
            yield event.plain_result(f"✅ 已删除预设: {key_to_delete}")
        else:
            yield event.plain_result(f"❌ 未找到名为 '{key_to_delete}' 的预设。")

    @filter.command("视频增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        args = event.message_str.strip().removeprefix("视频增加用户次数").strip()
        match = re.fullmatch(r"(\d+)\s+(\d+)", args)
        if not match: yield event.plain_result('格式错误: #视频增加用户次数 <QQ号> <次数>'); return
        target_qq, count = match.group(1), int(match.group(2))
        current_count = self._get_user_count(target_qq)
        self.user_counts[target_qq] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("视频增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        args = event.message_str.strip().removeprefix("视频增加群组次数").strip()
        match = re.fullmatch(r"(\d+)\s+(\d+)", args)
        if not match: yield event.plain_result('格式错误: #视频增加群组次数 <群号> <次数>'); return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[target_group] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("视频清除缓存", prefix_optional=True)
    async def on_clear_cache(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        count = 0
        try:
            if not await aiofiles.os.path.isdir(self.plugin_data_dir):
                yield event.plain_result("ℹ️ 缓存目录不存在，无需清理。"); return
            for filename in await aiofiles.os.listdir(self.plugin_data_dir):
                if filename.startswith("sora_video_") and filename.endswith(".mp4"):
                    filepath = self.plugin_data_dir / filename
                    await aiofiles.os.remove(filepath)
                    count += 1
            yield event.plain_result(f"✅ 成功清除了 {count} 个临时视频文件。")
        except Exception as e:
            logger.error(f"[PlatoSora] 清除缓存时发生错误: {e}", exc_info=True)
            yield event.plain_result(f"❌ 清除缓存时发生错误，请查看后台日志。")

    # --- 权限与工具函数 ---
    def is_global_admin(self, event: AstrMessageEvent):
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    async def _get_api_key(self) -> Optional[str]:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    async def _check_permissions(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        if self.is_global_admin(event): return True, None
        sender_id, group_id = event.get_sender_id(), event.get_group_id()
        if sender_id in self.conf.get("user_blacklist", []): return False, None
        if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",[]): return False, None
        if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist",[]): return False, "抱歉，您不在本功能的使用白名单中。"
        user_limit_on = self.conf.get("enable_user_limit", True)
        group_limit_on = self.conf.get("enable_group_limit", False) and group_id
        has_user_permission = not user_limit_on or self._get_user_count(sender_id) > 0
        has_group_permission = not group_limit_on or self._get_group_count(group_id) > 0
        if group_id and not has_group_permission and not has_user_permission: return False, "❌ 本群次数与您的个人次数均已用尽。"
        if not group_id and not has_user_permission: return False, "❌ 您的使用次数已用完。"
        return True, None

    async def _download_video_async(self, url: str) -> Optional[str]:
        filename = f"sora_video_{uuid.uuid4()}.mp4"
        filepath = str(self.plugin_data_dir / filename)
        try:
            async with self.api_client.session.get(url, timeout=300) as resp:
                resp.raise_for_status()
                video_data = await resp.read()
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(video_data)
            return filepath
        except Exception as e:
            logger.error(f"下载视频失败: {e}")
            if await aiofiles.os.path.exists(filepath):
                await aiofiles.os.remove(filepath)
            return None

    async def terminate(self):
        if self.api_client: await self.api_client.terminate()
        logger.info("[PlatoSora] 插件已终止")
