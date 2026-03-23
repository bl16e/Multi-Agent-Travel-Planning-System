# 测试模块

## 运行测试

```bash
# 安装项目
pip install -e .

# 运行所有测试
pytest tests/

# 运行特定模块
pytest tests/test_zhongshu.py
pytest tests/test_menxia.py
pytest tests/test_workflow.py
pytest tests/test_bureaus.py
pytest tests/test_integration.py

# 运行LLM测试（较慢）
pytest tests/ -k "llm or integration"
```

## 测试覆盖

- `test_zhongshu.py`: 中书省行程生成测试（含LLM调用）
- `test_menxia.py`: 门下省审核测试（含LLM调用）
- `test_workflow.py`: 工作流编排测试
- `test_bureaus.py`: 六部执行测试
- `test_integration.py`: 完整工作流集成测试

