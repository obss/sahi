---
tags:
  - contributing
  - development
---

# 为 SAHI 做贡献

感谢您有兴趣为 SAHI 做贡献！本指南将帮助您快速上手。

## 设置开发环境

### 1. Fork 和克隆

```bash
git clone https://github.com/YOUR_USERNAME/sahi.git
cd sahi
```

### 2. 创建环境

我们推荐使用 Python 3.10 进行开发：

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate  # Windows 用户: .venv\Scripts\activate
```

### 3. 安装依赖

```bash
# 安装核心 + 开发依赖
uv sync --extra dev

# 如需测试特定模型，请安装其对应的依赖。
```

## 代码格式化

我们使用 `ruff` 进行代码格式化和 lint 检查。格式化代码：

```bash
# 检查格式
uv run ruff check .
uv run ruff format --check .

# 修复格式
uv run ruff check --fix .
uv run ruff format .
```

或使用便捷脚本：

```bash
# 检查格式
python scripts/format_code.py check

# 修复格式
python scripts/format_code.py fix
```

## 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行指定测试文件
uv run pytest tests/test_predict.py

# 运行并生成覆盖率报告
uv run pytest --cov=sahi
```

## 提交 Pull Request

1. 创建新分支：`git checkout -b feature-name`
2. 进行修改
3. 格式化代码：`python scripts/format_code.py fix`
4. 运行测试：`uv run pytest`
5. 提交清晰的 commit 信息：`git commit -m "Add feature X"`
6. 推送并创建 PR：`git push origin feature-name`

## CI 构建失败

如果 CI 构建因格式问题失败：

1. 查看 CI 输出，确认失败的 Python 版本
2. 使用该版本创建环境：

   ```bash
   uv venv --python 3.X  # 将 X 替换为 CI 中的版本号
   source .venv/bin/activate
   ```

3. 安装开发依赖：

   ```bash
   uv sync --extra dev
   ```

4. 修复格式：

   ```bash
   python scripts/format_code.py fix
   ```

5. 提交并推送更改

## 添加新模型支持

要添加对新检测框架的支持：

1. 在 `sahi/models/your_framework.py` 下创建新文件
2. 实现继承自 `DetectionModel` 的类
3. 在 `sahi/auto_model.py` 中将您的框架添加到 `MODEL_TYPE_TO_MODEL_CLASS_NAME`
4. 在 `tests/test_yourframework.py` 下添加测试
5. 在 `docs/notebooks/inference_for_your_framework.ipynb` 下添加示例 notebook
6. 更新 [`README.md`](../../README.md) 和 `docs/` 下的相关文档以包含您的新模型

请参考现有实现，如 `sahi/models/ultralytics.py`。

## 有问题？

如果您有任何疑问，欢迎[发起讨论](https://github.com/obss/sahi/discussions)！
