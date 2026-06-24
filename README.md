# Agentic RAG

一个基于 FastAPI、LangGraph、Qdrant、MySQL、MinIO 和 PostgreSQL Checkpoint 的 Agentic RAG 后端项目。

项目目标不是只做一个简单的“上传文档后问答”，而是围绕真实 RAG 应用需要的工程链路进行实现：用户鉴权、文档上传、异步解析入库、向量检索、BM25、混合召回、Rerank、多轮会话、查询路由、问题改写、检索充分性判断、会话消息持久化和 LangGraph 状态恢复。

## 当前状态

- 用户注册、登录、JWT 鉴权
- 文档上传、解析、切分、向量化、入库
- MinIO 存储原始文件
- MySQL 存储用户、文档、chunk、会话和消息
- Qdrant 存储向量
- BM25 关键词检索，支持进程内缓存和失效
- 向量检索 + BM25 候选合并 + Rerank
- LangGraph Agentic RAG 工作流
- PostgreSQL Checkpoint 保存 LangGraph 状态
- 多轮会话基于 `conversation_id` / `thread_id` 串联
- RAG 检索评估脚本与评估结果
- 单元测试覆盖核心链路

当前测试结果：

```text
67 passed, 1 warning
```

其中 warning 来自 Qdrant client 在单元测试环境下无法检查服务端版本，不影响测试结果。

## 技术栈

| 模块 | 技术 |
|---|---|
| Web 框架 | FastAPI |
| 工作流编排 | LangGraph |
| LLM / Embedding | OpenAI-compatible API / LangChain |
| 向量数据库 | Qdrant |
| 关键词检索 | rank-bm25 + jieba |
| 重排序 | DashScope-compatible Rerank API |
| 业务数据库 | MySQL + SQLAlchemy Async |
| 状态恢复 | PostgreSQL + LangGraph Checkpoint |
| 文件存储 | MinIO |
| 数据库迁移 | Alembic |
| 测试 | pytest / pytest-asyncio |
| 包管理 | uv |

## 系统架构

```mermaid
flowchart TD
    U["User / Swagger / Frontend"] --> API["FastAPI API"]
    API --> Auth["JWT Auth"]
    API --> MySQL["MySQL: users / documents / chunks / conversations"]
    API --> MinIO["MinIO: original files"]
    API --> Qdrant["Qdrant: vectors"]
    API --> PG["PostgreSQL: LangGraph checkpoints"]
    API --> LG["LangGraph RAG Workflow"]
    LG --> Router["Router: 是否需要检索"]
    Router -->|unneeded| Chat["Chat Node"]
    Router -->|need| Contextualize["Contextualize Node: 问题改写"]
    Contextualize --> Retrieval["Hybrid Retrieval"]
    Retrieval --> Assess["Assess Node: 检索是否足够"]
    Assess -->|sufficient| Generate["Generate Node"]
    Assess -->|insufficient 且未重写| Rewrite["Rewrite Node"]
    Rewrite --> Retrieval
    Assess -->|insufficient 且已重写| Reject["Reject Node"]
```

## RAG 查询流程

`POST /rag/query` 使用 LangGraph 编排，核心状态包括：

- `messages`：多轮对话消息
- `original_query`：用户原始问题
- `search_query`：用于检索的改写问题
- `candidates`：检索和重排后的候选 chunks
- `retrieved_content`：拼接后的检索上下文
- `need_retrieve`：结构化路由结果，`need` / `unneeded`
- `retrieval_sufficient`：结构化评估结果，`sufficient` / `insufficient`
- `answer`：最终回答

流程：

```text
用户输入
 -> router_query 判断是否需要检索
 -> 如果不需要检索，进入 chat_node
 -> 如果需要检索，进入 contextualize_node 改写问题
 -> retrieve_node 执行混合召回 + rerank
 -> assess_node 判断检索结果是否足够
 -> 足够则 generate_node 基于历史 + 原问题 + 检索上下文回答
 -> 不足则 rewrite_node 改写后再检索一次
 -> 仍不足则 reject_node 拒答
```

### 为什么区分 `original_query` 和 `search_query`

多轮对话中，用户可能会问：

```text
它有什么特点？
```

这类问题不适合直接检索。系统会结合历史消息，将其改写成更适合检索的独立问题，例如：

```text
Hybrid RAG 有什么特点？
```

检索阶段使用 `search_query`，生成阶段使用：

```text
历史对话 + 用户原始问题 + 检索上下文
```

这样既保证检索准确性，也保留用户真实表达。

## 检索策略

当前实现包含三层能力：

1. 向量检索
   - 使用 embedding 模型生成向量
   - Qdrant 保存并召回相似 chunks

2. BM25 检索
   - 使用 `jieba` 分词
   - 使用 `rank-bm25` 构建关键词检索
   - 内置进程级 LRU 风格缓存
   - 文档上传、重试、删除后会失效对应缓存

3. 候选合并 + Rerank
   - 向量检索和 BM25 分别取候选
   - 合并去重
   - 调用 rerank 模型重排
   - 返回最终 Top K chunks

## 评估结果

评估集位于：

```text
evaluation/dataset.jsonl
```

当前主要评估结果位于：

```text
evaluation/results/v2_42q/
```

评估集规模：

```text
42 个可回答问题
```

主要结果：

| 方法 | Hit@5 | Recall@5 | MRR@5 | P95 延迟 |
|---|---:|---:|---:|---:|
| Vector baseline | 64.29% | 55.36% | 0.4587 | 408.15 ms |
| BM25 | 78.57% | 70.83% | 0.6040 | 183.16 ms |
| Hybrid RRF | 71.43% | 64.29% | 0.5563 | 619.63 ms |
| Union candidates + Rerank | 88.10% | 79.17% | 0.7258 | 2252.97 ms |

当前最优方案是：

```text
Vector Top30 + BM25 Top30 -> 候选合并 -> Rerank Top5
```

它提升了 Top5 命中率和排序质量，但代价是 rerank 延迟明显增加。

## API 概览

启动后可访问：

```text
http://127.0.0.1:8000/docs
```

### 用户接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/user/register` | 用户注册 |
| POST | `/user/login/username` | 用户名密码登录 |
| POST | `/user/login/phone` | 手机号密码登录 |

### RAG 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/rag/documents` | 上传文档 |
| POST | `/rag/documents/{document_id}/retry` | 重试失败文档 |
| GET | `/rag/documents` | 获取当前用户文档列表 |
| DELETE | `/rag/documents/{document_id}` | 删除文档 |
| POST | `/rag/query` | RAG 问答 |

### 查询请求示例

```json
{
  "user_query": "Hybrid RAG 有什么特点？",
  "conversation_id": null,
  "document_id": 2,
  "candidate_k": 30,
  "final_k": 5,
  "score_threshold": 0
}
```

响应示例：

```json
{
  "conversation_id": "9d146823-1b17-41e2-a82c-aa4db00c2290",
  "answer": "根据检索到的上下文，Hybrid RAG 的特点包括...",
  "sources": [
    {
      "filename": "LangChain-中文教程.pdf",
      "page_number": 130,
      "score": 0.59
    }
  ]
}
```

多轮追问时，需要复用第一次返回的 `conversation_id`：

```json
{
  "user_query": "它和普通 RAG 有什么区别？",
  "conversation_id": "9d146823-1b17-41e2-a82c-aa4db00c2290",
  "document_id": 2,
  "candidate_k": 30,
  "final_k": 5,
  "score_threshold": 0
}
```

## 环境变量

项目需要以下环境变量。不要提交真实 `.env` 文件。

```env
# MySQL
ASYNC_DATABASE_URL=mysql+aiomysql://user:password@localhost:3306/agentic_rag

# PostgreSQL for LangGraph checkpoint
LANGGRAPH_POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable

# Qdrant
QDRANT_ENDPOINT=http://localhost:6333
COLLECTION_NAME=rag_chunks

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-documents
MINIO_SECURE=false

# Model
API_KEY=your-api-key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=your-chat-model
EMBEDDING_MODEL=your-embedding-model
DIMENSIONS=512
RERANK_MODEL=qwen3-rerank
```

具体变量名以代码中的配置读取为准。

## 本地启动

### 1. 安装依赖

```powershell
uv sync
```

或：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. 启动依赖服务

需要提前启动：

- MySQL
- PostgreSQL
- Qdrant
- MinIO

如果使用 Docker，可以通过 `docker ps` 确认服务状态：

```bash
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"
```

PostgreSQL 中需要存在 LangGraph 使用的数据库，例如：

```sql
CREATE DATABASE langgraph;
```

LangGraph checkpoint 表会在应用启动时由 `checkpointer.setup()` 自动创建。

### 3. 执行数据库迁移

```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

检查迁移是否与模型一致：

```powershell
.\.venv\Scripts\alembic.exe check
```

### 4. 启动服务

```powershell
.\.venv\Scripts\uvicorn.exe main:app --reload
```

访问：

```text
http://127.0.0.1:8000/docs
```

## 测试

运行全量测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

当前结果：

```text
67 passed, 1 warning
```

重点测试文件：

| 文件 | 说明 |
|---|---|
| `test/test_workflow.py` | LangGraph 工作流分支测试 |
| `test/test_query.py` | `/rag/query` 接口逻辑测试 |
| `test/test_fusion.py` | 候选合并与 Rerank 流程测试 |
| `test/test_bm25_cache.py` | BM25 缓存测试 |
| `test/test_conversation_crud.py` | 会话 CRUD 测试 |
| `test/test_upload.py` | 文档上传测试 |
| `test/test_retry.py` | 文档重试测试 |
| `test/test_delete.py` | 文档删除测试 |

## 项目目录

```text
.
├── apps
│   ├── rag                 # RAG 上传、检索、问答、工作流
│   ├── user                # 用户注册、登录
│   └── conversations       # 会话 CRUD
├── config                  # MySQL / MinIO / Qdrant / Model 配置
├── models                  # SQLAlchemy ORM
├── alembic                 # 数据库迁移
├── evaluation              # RAG 评估脚本、数据集、结果
├── test                    # 单元测试
├── utils                   # JWT、安全工具
└── main.py                 # FastAPI 入口
```

## 已知限制与后续优化

当前项目已经具备完整 RAG 应用主链路，但仍有一些可优化点：

- 路由、改写、检索充分性判断的 prompt 还可以继续通过真实样例调优
- Rerank 提升明显，但延迟较高，需要根据场景做开关或异步优化
- BM25 当前使用进程内缓存，服务多实例部署时需要考虑外部检索服务或共享索引
- 会话接口可以继续补充列表、详情、标题生成、删除等前端友好能力
- 可增加 OpenTelemetry / structured logging，方便观察每个 LangGraph 节点耗时和决策
- 可继续扩展 SQLAgent、联网查询、Tool Calling，但不属于当前已完成能力

## 简历描述参考

可以概括为：

```text
基于 FastAPI + LangGraph 构建 Agentic RAG 后端系统，实现文档上传解析、向量化入库、BM25 + 向量混合召回、Rerank 重排、多轮会话记忆、查询路由、上下文问题改写、检索充分性判断和 PostgreSQL Checkpoint 状态恢复；基于 42 条评估集对 Vector、BM25、Hybrid、Rerank 方案进行离线评估，最佳方案 Hit@5 达到 88.10%，MRR@5 达到 0.7258。
```
