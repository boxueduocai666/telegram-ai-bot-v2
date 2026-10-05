# Telegram AI Bot V2

一个面向长期维护的个人 Telegram AI Bot。V2 在保留原有核心能力的基础上，对消息处理、AI Provider、联网搜索、视觉、图片搜索/生成、群聊总结、历史数据、SQLite 持久化与 Webhook 部署进行了整理。

V1 参考仓库：<https://github.com/boxueduocai666/telegram-ai-bot-v1>

## 1. 当前功能

- 私聊 AI 多轮对话
- 群聊 `@机器人` 触发、回复机器人消息触发
- 群聊短期共享上下文
- 回复/引用消息上下文理解
- 图片理解：直接发图或回复图片提问
- OpenAI-compatible AI 接口
- Agnes / Gemini Provider 与多模型切换
- `/model` Inline Keyboard 模型选择
- 用户模型选择 SQLite 持久化，重启后仍可保留
- `/clear` 清除当前对话上下文
- `/search <关键词>` 联网搜索，DDGS 主流程，可选 SearXNG fallback
- `/imagesearch <内容>` 搜索互联网图片
- `/image <描述>` AI 图片生成
- `/image 2.1 <描述>` 使用 Image 2.1 图片模型生成
- `/summary` 群聊 AI 总结
- 每群独立的自动总结设置与时区
- `/history` 历史上的今天
- `/history auto` 自动历史推送与时区设置
- `/status`、`/ping`、`/about`
- Markdown 到 Telegram MarkdownV2 的安全格式化，并在失败时自动降级纯文本
- Telegram Webhook + `/health`
- SQLite 持久化

## 2. 项目结构

```text
telegram-ai-bot-v2/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── handlers.py
│   ├── ai.py
│   ├── search.py
│   ├── image_search.py
│   ├── image_generation.py
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

Telegram 编排主要集中在 `handlers.py`，AI、搜索、视觉、图片能力、总结和数据库分别由独立模块负责，没有重新建立巨型 `bot_logic.py`。

## 3. 环境变量

本地使用 `.env`，部署平台则填写对应环境变量。主要配置包括：

- `TELEGRAM_BOT_TOKEN`
- `WEBHOOK_SECRET`
- `AI_API_KEY`
- `AI_BASE_URL`
- `DEFAULT_MODEL`
- `AVAILABLE_MODELS`
- Gemini 相关 API Key（启用 Gemini 时）
- `DATABASE_PATH`，持久化部署推荐 `/data/bot.db`
- `PUBLIC_URL`

不要把真实 Token、API Key、Secret 写入代码、README 或 Git。

## 4. AI Provider 与模型切换

V2 的文本 AI 以 OpenAI-compatible 接口为基础，并支持在 Bot 内选择不同 Provider 的模型。

使用：

```text
/model
```

先选择 Provider，再选择具体模型。当前用户选择会保存到 SQLite；Bot 运行期间也会保留在内存中，因此切换后不会出现“只回答一条消息，下一条又自动变回默认模型”的情况。

默认模型由：

```text
DEFAULT_MODEL
```

控制。可选模型由配置中的模型列表控制。

Gemini 模型需要正确配置对应的 Gemini API Key；未配置时，Bot 会拒绝切换到 Gemini，而不会把请求伪装成可用。

## 5. 联网搜索

`app/search.py` 使用 `ddgs` 执行搜索，可选 SearXNG 作为 fallback。

流程：

```text
Telegram
  ↓
handlers.py
  ↓
search.py
  ↓
DDGS / SearXNG
  ↓
搜索结果
  ↓
AI 整理
```

搜索失败只影响本次搜索，不应让主 Bot 崩溃，也不会伪造“已经查到网页”。

## 6. 图片能力

### 图片理解

`vision.py` 负责 Telegram 图片下载和视觉请求。

支持：

- 直接发送图片并附带问题
- 回复一张图片后提问

### 图片搜索

```text
/imagesearch 关键词
```

通过 DDGS 获取图片结果，并提供“点击查看图片”按钮。

### AI 图片生成

```text
/image 描述
/image 2.1 描述
```

图片生成使用独立的 `image_generation.py`，不会改变普通文本对话的模型选择逻辑。

## 7. 群聊

Bot 默认不会回答群里的每一条消息。

正常触发方式：

```text
@Bot 你好
```

或者：

```text
用户 → 回复 Bot → 提问
```

群聊普通消息会进入有限长度的短期上下文和总结缓冲区，因此 Bot 可以理解类似：

```text
A：这个软件今天更新了。
B：它现在好用吗？ @Bot
```

上下文会受到长度限制，不会无限增长。

## 8. `/summary` 群聊总结

群聊使用：

```text
/summary
```

总结群内近期有效消息，并尽量忽略“哈哈”“收到”“好的”等无实际信息的内容。

自动总结也可以按群独立配置，包括是否开启、总结时间、时区和最近一次发送状态。

## 9. `/history` 历史上的今天

支持：

```text
/history
/history 8月8日
/history 2008-08-08
/history 2008年8月8日
```

也支持群管理员配置自动推送：

```text
/history auto
/history auto on
/history auto off
/history auto 08:00
/history timezone Asia/Shanghai
```

历史资料优先使用 Wikimedia / Wikipedia 的 On This Day 数据源，而不是让 AI 自己凭记忆编造事件。数据源不可用时只对历史功能温和降级，不影响 AI、搜索或 Webhook 主流程。

## 10. Webhook 与部署

V2 使用 Webhook 模式，不使用 polling，也不调用 `getUpdates`。

健康检查：

```text
GET /health
```

Webhook：

```text
POST /telegram/webhook
```

Webhook 请求需要携带 Telegram Secret Token Header。

SQLite 数据库路径通过 `DATABASE_PATH` 配置。使用持久化磁盘/Volume 时建议：

```text
DATABASE_PATH=/data/bot.db
```

## 11. Markdown

`utils.py` 集中处理 Markdown 到 Telegram MarkdownV2 的转换，覆盖常用的：

- 标题
- 粗体
- 斜体
- 删除线
- 列表
- 引用
- 行内代码
- 多行代码
- 链接

MarkdownV2 发送失败时会自动尝试纯文本，避免因为格式字符导致整条回答发送失败。

## 12. 持久化与上下文

SQLite 主要用于保存需要跨重启保留的数据，例如：

- 用户选择的 AI 模型
- 群自动总结配置
- 群自动历史推送配置
- 群消息持久化记录

短期 AI 对话上下文仍主要保存在内存中。Bot 重启后，短期上下文会清空，但需要持久化的配置不会因此丢失。

## 13. 测试

项目包含基础测试：

```text
tests/test_ai.py
tests/test_search.py
tests/test_utils.py
```

建议在修改核心模块后运行：

```bash
pytest -q
```

## 14. 设计目标

V2 的目标不是堆叠越来越多的功能，而是在个人长期使用场景下保持：

- 简单
- 稳定
- 容易维护
- 模块职责清晰
- 出错时尽量局部降级
- 不因为一个附加功能故障而拖垮整个 Bot

V2 会继续在现有架构上进行受控改进，而不是为了功能数量不断重写整个项目。

## 15. 安全与隐私

- 密钥全部来自环境变量
- 日志不打印完整 Token / API Key / Secret
- Webhook 使用 Secret Token 校验
- `/status`、`/about` 不返回敏感配置
- 图片、群聊消息和引用内容可能被发送给第三方 AI Provider
- 使用 Bot 的群聊成员应了解相关数据可能被发送给所配置的 AI 服务
