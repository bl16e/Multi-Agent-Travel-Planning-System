# Multi-Agent Travel Planning System

基于LangGraph的多智能体旅行规划系统，采用"三省六部"架构设计。

## 系统架构

- **尚书省 (Orchestrator)**: 总协调者，管理整体工作流
- **中书省 (Itinerary)**: 行程规划智能体
- **门下省 (Review)**: 审核智能体
- **六部 (Bureaus)**:
  - 天气局 (Weather)
  - 预算局 (Budget)
  - 住宿局 (Accommodation)
  - 交通局 (Flight/Transport)
  - 日历局 (Calendar)

## 功能特性

- **可视化Web界面** - 实时展示"三省六部"工作流执行进度
- **智能行程规划** - 基于LangGraph的多智能体协作
- **多层审核机制** - 门下省审核确保方案质量
- **流式反馈** - Server-Sent Events实时推送进度日志
- **人工干预支持** - 在关键节点可介入调整
- **多格式输出** - Markdown旅行计划 + iCalendar日历文件

## 安装

### 环境要求

- Python 3.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 复制环境变量模板:
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的API密钥:
```
QWEN_API_KEY=your_qwen_api_key
QWEN_MODEL=qwen-plus-2025-07-28
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AMAP_API_KEY=your_amap_api_key
SERPAPI_API_KEY=your_serpapi_key
OUTPUT_DIR=artifacts
```

## 使用方法

### 命令行演示

```bash
python main.py
```

### Web界面 (推荐)

启动FastAPI服务器:
```bash
uvicorn main:app --reload
```

访问 http://localhost:8000 使用可视化界面:
- 填写旅行需求表单
- 实时查看工作流执行进度
- 可视化展示"三省六部"协作过程
- 查看完整旅行方案并下载

### API端点

- `POST /plan` - 创建旅行计划
- `POST /plan/stream` - 流式创建旅行计划
- `POST /resume/{request_id}` - 恢复并继续规划
- `GET /dashboard/{request_id}` - 查看规划状态
- `GET /download/{request_id}` - 下载生成的文件

## 项目结构

```
project/
├── main.py                 # FastAPI应用入口
├── workflow.py             # LangGraph工作流定义
├── provinces/              # 智能体模块
│   ├── shangshu_orchestrator/
│   ├── zhongshu_itinerary/
│   ├── menxia_review/
│   └── liubu/
├── utils/                  # 工具函数
├── templates/              # HTML模板
├── static/                 # 静态资源
└── artifacts/              # 输出文件目录
```

## 开发

运行测试:
```bash
pytest
```

## License

MIT
