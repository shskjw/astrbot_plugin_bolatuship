# astrbot_plugin_bolatuship

Sora 视频生成插件，支持柏拉图 API 和 OpenAI 兼容格式，支持 LLM 智能判断。

## 功能特性

- **文生视频**: 根据文字描述生成视频
- **图生视频**: 将图片转换为视频
- **多 API 支持**: 
  - 柏拉图原生 API
  - OpenAI 兼容格式 API
- **LLM 智能判断**: 自动识别用户意图，通过 LLM 工具调用生成视频
- **次数管理**: 用户/群组次数限制、签到奖励
- **预设系统**: 支持自定义预设提示词

## 安装

在 AstrBot 插件市场搜索 `bolatuship` 或手动安装：

```bash
# 克隆到插件目录
git clone https://github.com/shkjw/astrbot_plugin_bolatuship.git
```

## 配置说明

### API 模式

插件支持两种 API 模式：

#### 1. 柏拉图 API（默认）

```json
{
    "api_mode": "plato",
    "api_url": "https://api.bltcy.ai",
    "api_keys": ["your-api-key"],
    "default_model": "sora-2-fast"
}
```

#### 2. OpenAI 兼容格式

```json
{
    "api_mode": "openai",
    "openai_api_url": "https://your-api.com/v1/chat/completions",
    "api_keys": ["your-api-key"],
    "openai_model": "sora-2"
}
```

### LLM 工具配置

启用 LLM 工具后，AI 可以通过工具调用自动生成视频：

```json
{
    "enable_llm_tool": true,
    "llm_show_progress": true
}
```

## 使用方法

### 指令模式

```
# 文生视频
#生成视频 一只猫在草地上奔跑

# 图生视频（发送图片后）
#生成视频 让图片动起来

# 使用预设
#电影感 一个人走在雨中

# 添加参数
#生成视频 海浪拍打沙滩 (竖屏, 15s, 高清)
```

### LLM 模式

直接与 AI 对话：
- "帮我生成一个视频，内容是日落时分的海边"
- "把这张图片变成视频"
- "生成一段猫咪玩耍的视频"

### 管理指令

```
#视频签到          - 每日签到获取次数
#视频查询次数      - 查看剩余次数
#视频帮助          - 显示帮助信息
#视频预设列表      - 查看所有预设

# 管理员指令
#视频添加预设 电影感:cinematic, epic, 4k
#视频删除预设 电影感
#视频增加用户次数 123456 10
#视频增加群组次数 123456 10
#视频清除缓存
#视频今日统计
```

## 参数说明

在提示词末尾使用括号添加参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| 竖屏/9:16 | 竖屏比例 | `(竖屏)` |
| 横屏/16:9 | 横屏比例 | `(横屏)` |
| 高清/hd | 高清模式 | `(高清)` |
| Xs/X秒 | 视频时长 | `(15s)` |

组合使用：`(竖屏, 15s, 高清)`

## 项目结构

```
astrbot_plugin_bolatuship/
├── main.py              # 主插件文件
├── api_manager.py       # API 管理器（支持多种 API 格式）
├── data_manager.py      # 数据管理器（次数、签到、预设）
├── context_manager.py   # 上下文管理器（LLM 智能判断）
├── _conf_schema.json    # 配置文件模板
├── metadata.yaml        # 插件元数据
└── README.md            # 说明文档
```

## 更新日志

### v2.0.0
- 重构项目架构，采用分层设计
- 新增 OpenAI 兼容格式 API 支持
- 新增 LLM 工具调用功能
- 新增 LLM 智能判断功能
- 新增上下文管理器
- 优化代码结构和错误处理

### v1.0.1
- 初始版本
- 支持柏拉图 API
- 支持文生视频和图生视频
- 支持次数管理和签到

## 许可证

MIT License

## 作者

shskjw
