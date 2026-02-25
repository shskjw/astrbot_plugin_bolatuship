"""
数据管理器 - 处理用户数据、次数、签到等持久化存储
"""

import json
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from astrbot.api import logger


def norm_id(id_val: Any) -> str:
    """标准化 ID 为字符串"""
    if id_val is None:
        return ""
    return str(id_val).strip()


class DataManager:
    """数据管理器 - 处理持久化数据"""
    
    def __init__(self, data_dir: Path, config: Any):
        self.data_dir = data_dir
        self.config = config

        # 数据文件路径
        self.user_counts_file = self.data_dir / "video_user_counts.json"
        self.group_counts_file = self.data_dir / "video_group_counts.json"
        self.user_checkin_file = self.data_dir / "video_user_checkin.json"
        self.daily_stats_file = self.data_dir / "video_daily_stats.json"
        self.user_prompts_file = self.data_dir / "video_user_prompts.json"

        # 确保数据目录存在
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # 内存数据
        self.user_counts: Dict[str, int] = {}
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_data: Dict[str, str] = {}
        self.daily_stats: Dict[str, Any] = {}
        self.user_prompts: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}

    async def initialize(self):
        """初始化数据管理器"""
        await self._load_json(self.user_counts_file, "user_counts")
        await self._load_json(self.group_counts_file, "group_counts")
        await self._load_json(self.user_checkin_file, "user_checkin_data")
        await self._load_json(self.user_prompts_file, "user_prompts")

        if not self.daily_stats_file.exists():
            self.daily_stats = {"date": "", "users": {}, "groups": {}}
        else:
            await self._load_json(self.daily_stats_file, "daily_stats")

        self.reload_prompts()
        logger.info(f"[VideoGen] 数据管理器初始化完成，加载了 {len(self.prompt_map)} 个预设")

    async def _load_json(self, file_path: Path, attr_name: str):
        """加载 JSON 文件到属性"""
        if not file_path.exists():
            return
        try:
            content = await asyncio.to_thread(file_path.read_text, "utf-8")
            setattr(self, attr_name, json.loads(content))
        except Exception as e:
            logger.error(f"加载 {file_path} 失败: {e}")

    async def _save_json(self, file_path: Path, data: Any):
        """保存数据到 JSON 文件"""
        try:
            content = json.dumps(data, indent=4, ensure_ascii=False)
            await asyncio.to_thread(file_path.write_text, content, "utf-8")
        except Exception as e:
            logger.error(f"保存 {file_path} 失败: {e}")

    def reload_prompts(self):
        """重新加载预设提示词"""
        self.prompt_map.clear()

        # 从配置中加载 prompt_list
        prompt_list = self.config.get("prompt_list", [])
        if isinstance(prompt_list, list):
            for item in prompt_list:
                if ":" in item:
                    k, v = item.split(":", 1)
                    self.prompt_map[k.strip()] = v.strip()

        # 用户自定义预设（优先级最高）
        for k, v in self.user_prompts.items():
            self.prompt_map[k] = v

    def get_prompt(self, key: str) -> Optional[str]:
        """获取预设提示词"""
        return self.prompt_map.get(key)

    async def add_user_prompt(self, key: str, prompt: str):
        """添加或更新用户预设"""
        self.user_prompts[key] = prompt
        await self._save_json(self.user_prompts_file, self.user_prompts)
        self.reload_prompts()

    async def delete_user_prompt(self, key: str) -> bool:
        """删除用户预设"""
        if key in self.user_prompts:
            del self.user_prompts[key]
            await self._save_json(self.user_prompts_file, self.user_prompts)
            self.reload_prompts()
            return True
        return False

    # ================= 次数管理 =================

    def get_user_count(self, uid: str) -> int:
        """获取用户剩余次数"""
        return self.user_counts.get(norm_id(uid), 0)

    async def decrease_user_count(self, uid: str, amount: int = 1):
        """减少用户次数"""
        uid = norm_id(uid)
        count = self.get_user_count(uid)
        if amount <= 0 or count <= 0:
            return
        self.user_counts[uid] = count - min(amount, count)
        await self._save_json(self.user_counts_file, self.user_counts)

    async def add_user_count(self, uid: str, amount: int):
        """增加用户次数"""
        uid = norm_id(uid)
        self.user_counts[uid] = self.get_user_count(uid) + amount
        await self._save_json(self.user_counts_file, self.user_counts)

    def get_group_count(self, gid: str) -> int:
        """获取群组剩余次数"""
        return self.group_counts.get(norm_id(gid), 0)

    async def decrease_group_count(self, gid: str, amount: int = 1):
        """减少群组次数"""
        gid = norm_id(gid)
        count = self.get_group_count(gid)
        if amount <= 0 or count <= 0:
            return
        self.group_counts[gid] = count - min(amount, count)
        await self._save_json(self.group_counts_file, self.group_counts)

    async def add_group_count(self, gid: str, amount: int):
        """增加群组次数"""
        gid = norm_id(gid)
        self.group_counts[gid] = self.get_group_count(gid) + amount
        await self._save_json(self.group_counts_file, self.group_counts)

    # ================= 签到功能 =================

    async def process_checkin(self, uid: str) -> str:
        """处理签到"""
        uid = norm_id(uid)
        today = datetime.now().strftime("%Y-%m-%d")
        
        if self.user_checkin_data.get(uid) == today:
            return f"您今天已经签到过了！\n剩余次数: {self.get_user_count(uid)}"

        # 计算奖励
        reward = int(self.config.get("checkin_fixed_reward", 3))
        if str(self.config.get("enable_random_checkin", False)).lower() == 'true':
            max_r = int(self.config.get("checkin_random_reward_max", 5))
            reward = random.randint(1, max(1, max_r))

        await self.add_user_count(uid, reward)
        self.user_checkin_data[uid] = today
        await self._save_json(self.user_checkin_file, self.user_checkin_data)
        
        return f"🎉 签到成功！获得 {reward} 次，当前剩余: {self.get_user_count(uid)} 次。"

    # ================= 统计功能 =================

    async def record_usage(self, uid: str, gid: Optional[str]):
        """记录使用统计"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_stats.get("date") != today:
            self.daily_stats = {"date": today, "users": {}, "groups": {}}

        uid = norm_id(uid)
        self.daily_stats["users"][uid] = self.daily_stats["users"].get(uid, 0) + 1
        
        if gid:
            gid = norm_id(gid)
            self.daily_stats["groups"][gid] = self.daily_stats["groups"].get(gid, 0) + 1
        
        await self._save_json(self.daily_stats_file, self.daily_stats)

    def get_daily_stats(self) -> Dict[str, Any]:
        """获取今日统计"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_stats.get("date") != today:
            return {"date": today, "users": {}, "groups": {}}
        return self.daily_stats

    # ================= 权限检查 =================

    async def check_quota(
        self,
        uid: str,
        gid: Optional[str],
        is_admin: bool,
        cost: int = 1
    ) -> Dict[str, Any]:
        """
        检查用户配额
        
        Returns:
            {
                "allowed": bool,
                "source": "free" | "user" | "group" | None,
                "msg": str
            }
        """
        result = {"allowed": False, "source": None, "msg": ""}
        uid = norm_id(uid)
        gid = norm_id(gid) if gid else None

        # 1. 检查黑名单
        if uid in (self.config.get("user_blacklist") or []):
            result["msg"] = "❌ 您已被禁用此功能"
            return result
        if gid and gid in (self.config.get("group_blacklist") or []):
            result["msg"] = "❌ 该群组已被禁用此功能"
            return result

        # 2. 管理员始终允许
        if is_admin:
            result["allowed"] = True
            result["source"] = "free"
            return result

        # 3. 检查用户白名单
        user_whitelist = self.config.get("user_whitelist") or []
        if user_whitelist and uid not in user_whitelist:
            result["msg"] = "❌ 您不在白名单中，无权使用此功能"
            return result
        if user_whitelist and uid in user_whitelist:
            result["allowed"] = True
            result["source"] = "free"
            return result

        # 4. 检查群聊白名单
        group_whitelist = self.config.get("group_whitelist") or []
        if group_whitelist and gid and gid not in group_whitelist:
            result["msg"] = "❌ 该群组不在白名单中，无权使用此功能"
            return result
        if group_whitelist and gid and gid in group_whitelist:
            result["allowed"] = True
            result["source"] = "free"
            return result

        # 5. 检查次数限制
        enable_user_limit = self.config.get("enable_user_limit", True)
        enable_group_limit = self.config.get("enable_group_limit", False)
        
        if not enable_user_limit and not enable_group_limit:
            result["allowed"] = True
            result["source"] = "free"
            return result

        user_balance = self.get_user_count(uid)
        if enable_user_limit and user_balance >= cost:
            result["allowed"] = True
            result["source"] = "user"
            return result

        if gid and enable_group_limit:
            group_balance = self.get_group_count(gid)
            if group_balance >= cost:
                result["allowed"] = True
                result["source"] = "group"
                return result

        result["msg"] = f"❌ 次数不足 (需{cost}次)。用户剩余: {user_balance}"
        return result
