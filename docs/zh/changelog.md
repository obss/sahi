---
hide:
  - navigation
tags:
  - changelog
  - release-notes
---

# 📝 更新日志

## 🚀 SAHI v0.11.31 发布说明

我们很高兴地宣布 SAHI v0.11.31 版本，包含重要的 Bug 修复和改进！

## 🆕 更新内容

- 使 Category 不可变并添加测试 - @gboeer 在
  <https://github.com/obss/sahi/pull/1206>
- 更新 greedy_nmm 的文档字符串 - @kikefdezl 在
  <https://github.com/obss/sahi/pull/1205>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1208>

## 🙌 新贡献者

- @kikefdezl 在 <https://github.com/obss/sahi/pull/1205> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.30...0.11.31>

## 🚀 SAHI v0.11.30 发布说明

我们很高兴地宣布 SAHI v0.11.30 版本，改进了性能追踪、增强了测试基础设施，并提升了开发者体验！

### 📈 里程碑

- **引用 SAHI 的学术论文达到 400 篇！**
  ([#1168](https://github.com/obss/sahi/pull/1168))

### 🚀 关键更新

#### ⚡️ 性能与监控

- **修复了 `get_sliced_prediction` 中的后处理耗时追踪** -- 现在可以正确分离切片、预测和后处理的耗时，实现准确的性能监控
  ([#1201](https://github.com/obss/sahi/pull/1201)) - 感谢 @Toprak2！

#### 🧩 框架更新

- **重构 Ultralytics 支持**，增加 ONNX 模型支持和更好的兼容性
  ([#1184](https://github.com/obss/sahi/pull/1184))
- **更新 TorchVision 支持**至最新 API
  ([#1182](https://github.com/obss/sahi/pull/1182))
- **改进 Detectron2 支持**，优化配置处理以防止 KeyError 问题
  ([#1116](https://github.com/obss/sahi/pull/1116)) - 感谢 @Arnesh1411！
- **新增 Roboflow 框架支持**，用于 Roboflow Universe 的 RF-DETR 模型
  ([#1161](https://github.com/obss/sahi/pull/1161)) - 感谢 @nok！
- **移除 deepsparse 集成**，该框架已停止维护
  ([#1164](https://github.com/obss/sahi/pull/1164))

#### 🧪 测试基础设施

- **测试套件迁移至 pytest**
  ([#1187](https://github.com/obss/sahi/pull/1187))
  - 通过更好的并行执行加速测试
  - 扩展 Python 版本覆盖（3.8、3.9、3.10、3.11、3.12）
  - 更新至更新的 PyTorch 版本以实现更好的兼容性测试
  - 改进测试组织和可维护性
- **重构 MMDetection 测试**以提高可靠性
  ([#1185](https://github.com/obss/sahi/pull/1185))

#### 💻 开发者体验

- **新增 Context7 MCP 集成**，用于 AI 辅助开发
  ([#1198](https://github.com/obss/sahi/pull/1198))
  - SAHI 的文档现已在 [Context7 MCP](https://context7.com/obss/sahi) 中建立索引
  - 为 AI 编码助手提供最新的、版本特定的代码示例
  - 包含 [llms.txt](https://context7.com/obss/sahi/llms.txt) 文件，用于 AI 可读文档
  - 查看 [Context7 MCP 安装指南](https://github.com/upstash/context7#%EF%B8%8F-installation) 将 SAHI 文档集成到您的 AI 工作流程中

### 🛠️ 改进

#### 🧹 代码质量与安全性

- **不可变边界框**，用于线程安全操作
  ([#1194](https://github.com/obss/sahi/pull/1194),
  [#1191](https://github.com/obss/sahi/pull/1191)) - 感谢 @gboeer！
- **增强类型提示和文档字符串**
  ([#1195](https://github.com/obss/sahi/pull/1195)) - 感谢 @gboeer！
- **预测分数运算符重载**，支持直观的分数比较
  ([#1190](https://github.com/obss/sahi/pull/1190)) - 感谢 @gboeer！
- **PyTorch 现为软依赖**，提高灵活性
  ([#1162](https://github.com/obss/sahi/pull/1162)) - 感谢 @ducviet00！

### 🏗️ 基础设施与稳定性

- **改进依赖管理**和文档
  ([#1183](https://github.com/obss/sahi/pull/1183))
- **增强 pyproject.toml 配置**，实现更好的包管理
  ([#1181](https://github.com/obss/sahi/pull/1181))
- **优化 CI/CD 工作流**，用于 MMDetection 测试
  ([#1186](https://github.com/obss/sahi/pull/1186))

### 🐛 Bug 修复

- 修复 CUDA 设备选择，支持 cuda:0 以外的设备
  ([#1158](https://github.com/obss/sahi/pull/1158)) - 感谢 @0xf21！
- 修正参数命名，将 'confidence' 改为 'threshold' 以保持一致性
  ([#1180](https://github.com/obss/sahi/pull/1180)) - 感谢 @nok！
- 修复设备选择函数中的正则表达式格式
  ([#1165](https://github.com/obss/sahi/pull/1165))
- 解决未安装 PyTorch 时的 torch 导入错误
  ([#1172](https://github.com/obss/sahi/pull/1172)) - 感谢 @ducviet00！
- 修复 `AutoDetectionModel.from_pretrained` 的模型实例化问题
  ([#1158](https://github.com/obss/sahi/pull/1158))

### 📦 依赖项

- 将 OpenCV 包从 4.10.0.84 更新至 4.11.0.86
  ([#1171](https://github.com/obss/sahi/pull/1171)) - 感谢 @ducviet00-h2！
- 移除已停止维护的 matplotlib-stubs 依赖
  ([#1169](https://github.com/obss/sahi/pull/1169))
- 清理未使用的配置文件
  ([#1199](https://github.com/obss/sahi/pull/1199))

### 📚 文档

- 新增 context7.json 以改进 AI 工具集成
  ([#1200](https://github.com/obss/sahi/pull/1200))
- 更新 README 中的贡献者信息
  ([#1175](https://github.com/obss/sahi/pull/1175),
  [#1179](https://github.com/obss/sahi/pull/1179))
- 新增 Roboflow+SAHI Colab 教程链接
  ([#1177](https://github.com/obss/sahi/pull/1177))

### 🙏 致谢

特别感谢所有让此版本成为可能的贡献者：@nok、@gboeer、@Toprak2、@Arnesh1411、@0xf21、@ducviet00、@ducviet00-h2、@p-constant 和 @fcakyon！

---

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.24...0.11.30>

## 🚀 SAHI v0.11.29 发布说明

### 🆕 更新内容

- 使边界框不可变 - @gboeer 在
  <https://github.com/obss/sahi/pull/1194>
- 改进类型提示和文档字符串 - @gboeer 在
  <https://github.com/obss/sahi/pull/1195>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1196>

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.28...0.11.29>

## 🚀 SAHI v0.11.28 发布说明

### 🆕 更新内容

- 为预测分数添加运算符重载 - @gboeer 在
  <https://github.com/obss/sahi/pull/1190>
- 改进 detectron2 支持 - @Arnesh1411 在
  <https://github.com/obss/sahi/pull/1116>
- 为边界框使用不可变参数 - @gboeer 在
  <https://github.com/obss/sahi/pull/1191>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1192>

### 🙌 新贡献者

- @Arnesh1411 在 <https://github.com/obss/sahi/pull/1116> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.27...0.11.28>

## 🚀 SAHI v0.11.27 发布说明

## 🆕 更新内容

- 修复：将推理方法中的 'confidence' 更新为 'threshold' - @nok 在
  <https://github.com/obss/sahi/pull/1180>
- 更新 README.md - @nok 在 <https://github.com/obss/sahi/pull/1179>
- 改进 pyproject.toml - @fcakyon 在 <https://github.com/obss/sahi/pull/1181>
- 重构依赖管理和部分文档 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1183>
- 更新：重构 ultralytics 支持 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1184>
- 重构 mmdet 测试 - @fcakyon 在 <https://github.com/obss/sahi/pull/1185>
- 更新 torchvision 支持至最新 API - @fcakyon 在
  <https://github.com/obss/sahi/pull/1182>
- 优化 mmdet 工作流触发条件 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1186>
- 迁移测试至 pytest - @fcakyon 在
  <https://github.com/obss/sahi/pull/1187>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1188>

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.26...0.11.27>

## 🚀 SAHI v0.11.26 发布说明

### 🆕 更新内容

- 将 opencv 包从 `4.10.0.84` 升级至 `4.11.0.86` - @ducviet00-h2 在
  <https://github.com/obss/sahi/pull/1171>
- 新增 Roboflow 框架（RFDETR 模型）- @nok 在
  <https://github.com/obss/sahi/pull/1161>
- 将新贡献者添加到 readme - @fcakyon 在
  <https://github.com/obss/sahi/pull/1175>
- 将 roboflow+sahi colab 链接添加到 readme - @fcakyon 在
  <https://github.com/obss/sahi/pull/1177>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1176>

### 🙌 新贡献者

- @ducviet00-h2 在 <https://github.com/obss/sahi/pull/1171> 中首次贡献
- @nok 在 <https://github.com/obss/sahi/pull/1161> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.25...0.11.26>

## 🚀 SAHI v0.11.25 发布说明

### 🆕 更新内容

- 更新 readme 中的 sahi 引用 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1168>
- 移除已停止维护的 matplotlib-stubs - @fcakyon 在
  <https://github.com/obss/sahi/pull/1169>
- 修复 torch 导入错误 - @ducviet00 在
  <https://github.com/obss/sahi/pull/1172>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1173>

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.24...0.11.25>

## 🚀 SAHI v0.11.24 发布说明

### 🆕 更新内容

- 修复拼写错误和脚本 URL - @gboeer 在
  <https://github.com/obss/sahi/pull/1155>
- 修复 CI 工作流 bug - @Dronakurl 在 <https://github.com/obss/sahi/pull/1156>
- [文档] 修复拼写错误 - @gboeer 在 <https://github.com/obss/sahi/pull/1157>
- 移除 deepsparse 集成 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1164>
- 修复：使 PyTorch 不再是硬依赖 - @ducviet00 在
  <https://github.com/obss/sahi/pull/1162>
- 修复：支持指定 cuda:0 以外的设备 - @0xf21 在
  <https://github.com/obss/sahi/pull/1158>
- 修复：修正 select_device 函数中的正则表达式格式 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1165>
- 为 yolov8onnx 添加 TensorrtExecutionProvider - @p-constant 在
  <https://github.com/obss/sahi/pull/1091>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1166>

### 🙌 新贡献者

- @gboeer 在 <https://github.com/obss/sahi/pull/1155> 中首次贡献
- @ducviet00 在 <https://github.com/obss/sahi/pull/1162> 中首次贡献
- @0xf21 在 <https://github.com/obss/sahi/pull/1158> 中首次贡献
- @p-constant 在 <https://github.com/obss/sahi/pull/1091> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.23...0.11.24>

## 🚀 SAHI v0.11.23 发布说明

### 🆕 更新内容

- 修复(CI)：numpy 依赖修复 #1119 - @Dronakurl 在
  <https://github.com/obss/sahi/pull/1144>
- 修复：Predict 无法在源目录中找到 TIF 文件 - @dibunker 在
  <https://github.com/obss/sahi/pull/1142>
- 修复 demo Notebooks 中的拼写错误 - @picjul 在
  <https://github.com/obss/sahi/pull/1150>
- 修复：修复多边形修复和空多边形问题，见 #1118 - @mario-dg 在
  <https://github.com/obss/sahi/pull/1138>
- 改进包 CI 日志 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1151>

### 🙌 新贡献者

- @dibunker 在 <https://github.com/obss/sahi/pull/1142> 中首次贡献
- @picjul 在 <https://github.com/obss/sahi/pull/1150> 中首次贡献
- @mario-dg 在 <https://github.com/obss/sahi/pull/1138> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.22...0.11.23>

## 🚀 SAHI v0.11.22 发布说明

### 🆕 更新内容

- 改进对最新 mmdet (v3.3.0) 的支持 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1129>
- 改进对最新 yolov5-pip 和 ultralytics 版本的支持 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1130>
- 支持最新的 huggingface/transformers 模型 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1131>
- 重构 coco 到 yolo 的转换，更新文档 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1132>
- 升级版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/1134>

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.21...0.11.22>

### 📚 核心文档文件

#### 📦 [预测工具](predict.md)

- 执行目标检测推理的详细指南

* 标准和切片推理示例
* 批量预测用法
* 推理时的类别排除
* 可视化参数和导出格式
* 各模型集成的交互式示例（YOLOv8、MMDetection 等）

#### ✂️ [切片工具](slicing.md)

- 大图像和数据集切片指南

* 图像切片示例
* COCO 数据集切片示例
* 交互式示例 notebook 参考

#### 🐒 [COCO 工具](coco.md)

- 处理 COCO 格式数据集的全面指南

* 数据集创建和操作
* COCO 数据集切片
* 数据集拆分（训练集/验证集）
* 类别筛选和更新
* 基于面积的筛选
* 数据集合并
* 格式转换（COCO ↔ YOLO）
* 数据集采样工具
* 统计计算
* 结果验证

#### 💻 [CLI 命令](cli.md)

- SAHI 命令行界面完整参考

* 预测命令
* FiftyOne 集成
* COCO 数据集操作
* 环境信息
* 版本检查
* 自定义脚本用法

#### 👁️ [FiftyOne 集成](fiftyone.md)

- 使用 FiftyOne 可视化和分析预测的指南

* 数据集可视化
* 结果探索
* 交互式分析

#### 📓 交互式示例

所有文档文件都配有 [demo 目录](../../demo/) 中的交互式 Jupyter notebook：

- `slicing.ipynb` - 切片操作演示
- `inference_for_ultralytics.ipynb` - YOLOv8/YOLO11/YOLO12 集成
- `inference_for_yolov5.ipynb` - YOLOv5 集成
- `inference_for_mmdetection.ipynb` - MMDetection 集成
- `inference_for_huggingface.ipynb` - HuggingFace 模型集成
- `inference_for_torchvision.ipynb` - TorchVision 模型集成
- `inference_for_rtdetr.ipynb` - RT-DETR 集成
- `inference_for_sparse_yolov5.ipynb` - DeepSparse 优化推理

### 🚦 快速上手

如果您是 SAHI 新用户：

1. 从[预测工具](predict.md)开始，了解基本推理
2. 探索[切片工具](slicing.md)，学习处理大图像
3. 查看 [CLI 命令](cli.md)，了解命令行用法
4. 深入 [COCO 工具](coco.md)，学习数据集操作
5. 尝试 [demo 目录](../../demo/) 中的交互式 notebook，获取实践经验

## 🚀 SAHI v0.11.21 发布说明

### 🆕 更新内容

- 使用预训练或自定义模型在推理时排除类别 - @gguzzy 在
  <https://github.com/obss/sahi/pull/1104>
- pyproject.toml、pre-commit、ruff、uv 和类型问题修复 #1119 - @Dronakurl 在
  <https://github.com/obss/sahi/pull/1120>
- 在 predict 文档中添加类别排除示例 - @gguzzy 在
  <https://github.com/obss/sahi/pull/1125>
- 添加 OBB 示例 - @fcakyon 在 <https://github.com/obss/sahi/pull/1126>
- 修复 predict 函数中的类型提示拼写错误 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1111>
- 移除 numpy<2 上限限制 - @weiji14 在
  <https://github.com/obss/sahi/pull/1112>
- 修复 readme 中的 CI 徽章 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1124>
- 修复 pyproject.toml 中的版本 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1127>

### 🙌 新贡献者

- @Dronakurl 在 <https://github.com/obss/sahi/pull/1120> 中首次贡献
- @gguzzy 在 <https://github.com/obss/sahi/pull/1104> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.20...0.11.21>

## 🚀 SAHI v0.11.20 发布说明

### 🆕 更新内容

- 添加 yolo11 和 ultralytics OBB 任务支持 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1109>
- 支持最新 opencv 版本 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1106>
- 简化 yolo 检测模型代码 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1107>
- 固定 shapely>2.0.0 - @weiji14 在 <https://github.com/obss/sahi/pull/1101>

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.19...0.11.20>

## 🚀 SAHI v0.11.19 发布说明

### 🆕 更新内容

- 修复 CI actions - @fcakyon 在 <https://github.com/obss/sahi/pull/1073>
- 更新 mmdet 模型的 has_mask 方法（处理边界情况）- @ccomkhj 在
  <https://github.com/obss/sahi/pull/1066>
- 处理另一个自相交的边界情况 - @sergiev 在
  <https://github.com/obss/sahi/pull/982>
- 更新 README.md - @fcakyon 在 <https://github.com/obss/sahi/pull/1077>
- 移除无法正常工作的 yolonas 支持 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1097>
- 移除 yolonas 支持（第二部分）- @fcakyon 在
  <https://github.com/obss/sahi/pull/1098>
- 更新 mmdet 模型的 has_mask 方法（处理 ConcatDataset）- @ccomkhj 在
  <https://github.com/obss/sahi/pull/1092>

### 🙌 新贡献者

- @ccomkhj 在 <https://github.com/obss/sahi/pull/1066> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.18...0.11.19>

## 🚀 SAHI v0.11.18 发布说明

### 🆕 更新内容

- 添加 yolov8 mask 支持，mask 处理速度提升 4-5 倍 - @mayrajeo 在
  <https://github.com/obss/sahi/pull/1039>
- 修复 mmdet 模型的 has_mask 方法 - @Alias-z 在
  <https://github.com/obss/sahi/pull/1054>
- 修复切片 COCO 时的 `TypeError: 'GeometryCollection' object is not subscriptable` 问题 - @Alias-z 在
  <https://github.com/obss/sahi/pull/1047>
- 支持 opencv-python 4.9 版本 - @iokarkan 在
  <https://github.com/obss/sahi/pull/1041>
- 为 numpy 依赖添加上限 - @fcakyon 在
  <https://github.com/obss/sahi/pull/1057>
- 添加更多单元测试 - @MMerling 在 <https://github.com/obss/sahi/pull/1048>
- 升级 CI actions - @fcakyon 在 <https://github.com/obss/sahi/pull/1049>

### 🙌 新贡献者

- @iokarkan 在 <https://github.com/obss/sahi/pull/1041> 中首次贡献
- @MMerling 在 <https://github.com/obss/sahi/pull/1048> 中首次贡献
- @Alias-z 在 <https://github.com/obss/sahi/pull/1047> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.16...0.11.18>

## 🚀 SAHI v0.11.16 发布说明

## 🚀 SAHI v0.11.15 发布说明

## 🚀 SAHI v0.11.14 发布说明

### 🆕 更新内容

- 支持 Deci-AI YOLO-NAS 模型 - @ssahinnkadir 在
  <https://github.com/obss/sahi/pull/874>
- Detectron2 模型的显著速度提升 - @MyosQ 在
  <https://github.com/obss/sahi/pull/865>
- 支持 ultralytics>=8.0.99 - @eVen-gits 在
  <https://github.com/obss/sahi/pull/873>
- 文档拼写错误和缺失值修复 - @Hamzalopode 在
  <https://github.com/obss/sahi/pull/859>
- 更新版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/876>
- 更新 black 版本 - @fcakyon 在 <https://github.com/obss/sahi/pull/877>

### 🙌 新贡献者

- @Hamzalopode 在 <https://github.com/obss/sahi/pull/859> 中首次贡献
- @eVen-gits 在 <https://github.com/obss/sahi/pull/873> 中首次贡献
- @MyosQ 在 <https://github.com/obss/sahi/pull/865> 中首次贡献

**完整更新日志**：<https://github.com/obss/sahi/compare/0.11.13...0.11.14>

## 🚀 SAHI v0.11.13 发布说明

## 🚀 SAHI v0.11.12 发布说明

## 🚀 SAHI v0.11.11 发布说明

## 🚀 SAHI v0.11.10 发布说明

## 🚀 SAHI v0.11.9 发布说明

## 🚀 SAHI v0.11.8 发布说明

## 🚀 SAHI v0.11.7 发布说明

## 🚀 SAHI v0.11.6 发布说明

## 🚀 SAHI v0.11.5 发布说明

## 🚀 SAHI v0.11.4 发布说明

## 🚀 SAHI v0.11.3 发布说明

## 🚀 SAHI v0.11.2 发布说明

## 🚀 SAHI v0.11.1 发布说明
