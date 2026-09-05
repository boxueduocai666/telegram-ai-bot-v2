# Telegram AI Bot V2

一个面向长期维护的个人 Telegram AI Bot。V2 不是对 V1 的简单打补丁，而是保留主要能力后重新整理消息处理、AI、搜索、视觉、总结、持久化和部署结构。

V1 参考仓库：<https://github.com/boxueduocai666/telegram-ai-bot-v1>

## 1. 功能

- 私聊多轮 AI 对话
- 群聊 `@机器人`、回复机器人、回复任意消息后提问
- 一层引用上下文：明确 Quote > 被回复消息 > 当前短期群聊上下文 > 更早历史
- 长引用截断，避免无限递归上下文
- 图片理解：直接发图或回复图片提问
- OpenAI-compatible AI Provider，可切换 Agnes / OpenAI / Qwen / DeepSeek / 其他兼容服务
- `/model` Inline Keyboard + SQLite 持久化
- `/search` 强制联网搜索，DDGS 主流程，可选 SearXNG fallback
- `/history` 历史上的今天：支持今天、月日、完整日期
- `/history auto on|off`、`/history auto HH:MM`、`/history timezone <IANA>`
- 每群独立时区与自动推送时间，默认 08:00、Asia/Shanghai、关闭
- `/summary` 群聊总结；群聊累计约 30 条有效消息后自动总结并清理摘要缓冲
- Markdown -> Telegram MarkdownV2 安全格式化；发送失败自动降级为纯文本
- Railway Webhook only + `/health`
- SQLite + Railway Volume

## 2. 项目结构

```text
telegram-ai-bot-v2/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── handlers.py
│   ├── ai.py
│   ├── search.py
│   ├── vision.py
│   ├── summary.py
│   ├── database.py
│   └── utils.py
├── data/.gitkeep
├── tests/
│   ├── test_ai.py
│   ├── test_search.py
│   └── test_utils.py
├── .env.example
├── .gitignore
├── requirements.txt
├── railway.toml
├── Dockerfile
├── README.md
└── LICENSE
```

没有重新建立 `bot_logic.py`。Telegram 编排集中在 `handlers.py`，能力模块分别负责自己的工作。

## 3. 环境变量

复制 `.env.example` 为 `.env`（本地）或在 Railway Variables 中填写：

- `TELEGRAM_BOT_TOKEN`
- `WEBHOOK_SECRET`
- `AI_API_KEY`
- `AI_BASE_URL`
- `DEFAULT_MODEL`
- `AVAILABLE_MODELS`
- `DATABASE_PATH`，Railway 推荐 `/data/bot.db`
- `PUBLIC_URL`（或 Railway 提供的公开域名变量）

不要把真实 Token、API Key、Secret 写进代码、README 或 Git。

## 4. 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

打开 `GET /health`，应返回 `ok`。

## 5. Railway 部署

### Variables

填写 `.env.example` 中的变量。

### Volume

创建一个 Railway Volume，并挂载到：

```text
/data
```

数据库使用：

```text
DATABASE_PATH=/data/bot.db
```

Railway Volume 用于持久化服务数据；应用启动时只要在挂载点下读写数据库即可。

### Webhook

设置：

```text
PUBLIC_URL=https://你的 Railway 服务域名
WEBHOOK_SECRET=随机长字符串
```

应用启动后只注册 Telegram Webhook，不创建 polling，也不调用 `getUpdates`。

Webhook 接口：

```text
POST /telegram/webhook
```

请求必须带：

```text
X-Telegram-Bot-Api-Secret-Token: <WEBHOOK_SECRET>
```

### Healthcheck

Railway Healthcheck：

```text
/health
```

## 6. AI Provider

V2 依赖 OpenAI-compatible Chat Completions 接口。核心代码只依赖：

```text
AI_API_KEY
AI_BASE_URL
DEFAULT_MODEL
```

更换 Provider 一般只需要修改配置，不需要改 Telegram handler。

视觉请求也通过同一个兼容客户端发送，因此所选模型必须具备相应视觉能力；如果某 Provider 的视觉模型需要特殊 API 格式，则该 Provider 需要在 `vision.py` / `ai.py` 中做针对性适配，而不能伪装成通用兼容能力。

## 7. 模型配置

例如：

```text
DEFAULT_MODEL=agnes-2.0-flash
AVAILABLE_MODELS=agnes-2.0-flash,agnes-2.5-flash,agnes-2.5-pro
```

`/model` 使用 Inline Keyboard。用户选中的模型写入 SQLite，Bot 重启后不会丢失。

## 8. 联网搜索

`app/search.py` 使用 `ddgs`。DDGS 本身负责聚合多个搜索后端，因此不要求你额外申请独立的搜索 API Key。

流程：

```text
Telegram
  ↓
handlers.py
  ↓
search.py
  ↓
DDGS
  ↓
搜索结果
  ↓
AI
```

搜索失败不会让主 Bot 崩溃，也不会伪造“已经查到网页”。当前实现支持通过环境变量扩展 SearXNG fallback，默认不启用。

## 9. 图片理解

`vision.py` 负责从 Telegram 获取最高分辨率图片、下载、Base64 Data URL、构造视觉请求。

支持：

- 直接发送图片 + 问题
- 回复图片 + 问题

如果视觉模型不可用，Bot 会给出功能级错误，而不是崩溃。

## 10. Markdown

`utils.py` 集中处理 Markdown -> Telegram MarkdownV2：

- 标题
- 粗体
- 斜体
- 删除线
- 列表
- 引用
- 行内代码
- 多行代码
- 链接
- 中文与中文标点

MarkdownV2 发送异常时自动尝试纯文本发送。

引用用户原文不会直接作为 Bot 的 Markdown 发送内容，因此不会因为用户原文里的特殊字符破坏输出格式。

## 11. `/summary`

群聊使用：

```text
/summary
```

总结最近已进入该群短期内存上下文的聊天内容。它不会为了每次总结把 SQLite 当作无限消息仓库，也不会无限读取群历史。

## 12. `/history`

```text
/history
/history 8月8日
/history 2008-08-08
/history 2008年8月8日
```

`/history` 默认按当前服务默认时区的当天查询；群自动推送则按群自身时区计算。

历史数据优先来自 Wikimedia / Wikipedia On This Day feed，而不是让 AI 自己凭记忆编事件。

Wikimedia 的 On This Day feed 当前属于实验性接口，且官方资料已经说明 Wikifeeds 正处于逐步退役路径。因此 V2 对该数据源采取“功能失败即温和降级”的策略：接口失败只影响 `/history`，不会影响 AI、搜索或 Webhook 主流程。未来 Wikimedia 发布替代 API 时，只需替换 `fetch_history_events()` 数据层。

## 13. 自动历史推送

默认关闭。

管理员：

```text
/history auto on
/history auto off
/history auto 08:00
/history timezone Asia/Shanghai
```

查看状态：

```text
/history auto
```

每个群保存自己的：

- 是否开启
- 推送时间
- IANA 时区
- 最近一次成功发送日期

定时任务是单进程内的 asyncio loop。Railway 推荐该服务保持单实例；多实例部署时应增加分布式锁/外部任务系统，否则同一个群可能被多个实例同时调度。

## 14. 群聊使用

Bot 不会处理所有群消息。

正常触发方式：

```text
@Bot 你好
```

回复 Bot：

```text
用户 → 回复 Bot → 问问题
```

回复任意消息：

```text
A：苹果发布了新产品。
B：回复 A：这是真的吗？
```

上下文优先级：

```text
明确 Quote
    ↓
被回复消息
    ↓
当前群短期上下文
    ↓
更早短期历史
```

只处理一层直接回复上下文，不递归追踪“回复的回复”，避免无限套娃。

## 15. 安全说明

- 所有密钥来自环境变量
- 日志不打印完整 Token / API Key / Secret
- `/status`、`/about` 不返回敏感信息
- Webhook 使用 Telegram Secret Token Header 校验
- 搜索结果与 AI 回答分离；搜索失败不伪造成功
- 图片、群聊消息和引用内容可能被发送给第三方 AI Provider，正式部署前应在群规则/隐私说明中明确告知使用者

## 16. 测试

运行：

```bash
pytest -q
```

当前测试重点覆盖：

- AI 调用
- 搜索空输入
- 搜索结果格式化
- 文本截断
- 时区
- MarkdownV2 转义

实际接入 Telegram / Provider 后，建议再增加基于 mock Update 的 handler 集成测试。

## 17. 已知边界与诚实说明

### 多实例

本项目的自动推送设计明确以 Railway 单实例为目标。SQLite 本身适合单服务持久化，但不应被当作多实例分布式锁服务。

### 上下文

短期 AI 上下文目前在内存，不会因为每条普通消息而永久写 SQLite。重启后短期聊天上下文会清空，但模型选择与群自动推送配置会保留。

### Markdown

这里实现的是针对 Telegram Bot 输出的实用 Markdown 子集，不是完整 CommonMark 解析器。即便解析失败，发送仍会自动回退纯文本。

### 视觉 Provider

“OpenAI-compatible”主要解决文本对话接口兼容；不同厂商视觉接口可能仍有差异。若目标 Provider 不支持 OpenAI 风格 `image_url` 消息，需要针对该 Provider 进行适配。
