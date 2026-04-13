# CLI 命令

SAHI 提供了一套全面的命令行工具用于目标检测任务。本指南涵盖所有可用命令，附有详细的示例和选项说明。

## `predict` 命令用法

对图像或视频执行切片推理，以更好地检测小目标。

### 基本用法

```bash
sahi predict --source image/file/or/folder --model_path path/to/model --model_config_path path/to/config
```

该命令会使用默认参数进行切片推理，并将预测可视化结果导出到 `runs/predict/exp`
文件夹。

### 视频输入支持

SAHI 支持使用相同的命令结构进行视频推理：

```bash
sahi predict --model_path yolo26s.pt --model_type ultralytics --source video.mp4
```

#### 实时视频可视化

使用 `--view_video` 参数在推理过程中查看视频渲染：

```bash
sahi predict --model_path yolo26s.pt --model_type ultralytics --source video.mp4 --view_video
```

**键盘控制：**

- **`D`** - 前进 100 帧
- **`A`** - 后退 100 帧
- **`G`** - 前进 20 帧
- **`F`** - 后退 20 帧
- **`Esc`** - 退出查看器

> **提示：** 如果 `--view_video` 运行较慢，可以添加 `--frame_skip_interval=20`
> 来跳过 20 帧的间隔。

### 高级切片参数

自定义切片行为以获得最佳检测效果：

```bash
sahi predict --slice_width 512 --slice_height 512 \
  --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 \
  --model_confidence_threshold 0.25 \
  --source image/file/or/folder \
  --model_path path/to/model \
  --model_config_path path/to/config
```

#### 模型配置

**检测框架：**

- `--model_type mmdet` - 用于 MMDetection 模型
- `--model_type ultralytics` - 用于 Ultralytics/YOLOv5/YOLO11 模型
- `--model_type huggingface` - 用于 HuggingFace 模型
- `--model_type torchvision` - 用于 Torchvision 模型

**置信度阈值：**

- `--model_confidence_threshold 0.25` - 设置检测的最低置信度

#### 后处理选项

**后处理类型：**

- `--postprocess_type GREEDYNMM` - 贪心非最大合并（默认）
- `--postprocess_type NMS` - 标准非最大抑制

**匹配指标：**

- `--postprocess_match_metric IOS` - 交集与较小面积之比
- `--postprocess_match_metric IOU` - 交并比（默认）

**其他选项：**

- `--postprocess_match_threshold 0.5` - 设置匹配阈值
- `--postprocess_class_agnostic` - 后处理时忽略类别 ID

#### 导出选项

**可视化导出：**

- `--novisual` - 禁用预测可视化导出
- `--visual_export_format JPG` - 设置导出格式（JPG、PNG 等）

**数据导出：**

- `--export_pickle` - 导出预测 pickle 文件
- `--export_crop` - 导出裁剪后的检测结果

#### 推理模式

默认情况下，SAHI 执行多阶段推理（同时进行标准推理和切片推理）：

- `--no_sliced_prediction` - 禁用切片推理（仅标准推理）
- `--no_standard_prediction` - 禁用标准推理（仅切片推理）

### COCO 数据集评估

使用 COCO 标注文件进行预测评估：

```bash
sahi predict --dataset_json_path dataset.json \
  --source path/to/coco/image/folder \
  --model_path path/to/model
```

预测结果会以 COCO JSON 格式导出到
`runs/predict/exp/results.json`。之后你可以使用：

- `sahi coco evaluate` - 计算 COCO 评估指标
- `sahi coco analyse` - 生成详细的误差分析图

### 进度报告

启用进度条来跟踪推理进度：

```bash
sahi predict --model_path path/to/model --source images/ \
  --slice_width 512 --slice_height 512 --progress_bar
```

> **注意：** `--progress_bar`
> 参数控制 CLI 的可视化进度（tqdm）。`progress_callback` 参数仅在 Python
> API 中可用，不作为 CLI 选项暴露。

---

## `predict-fiftyone` 命令用法

执行切片推理并使用 FiftyOne App 交互式可视化结果。

### 基本用法

```bash
sahi predict-fiftyone --image_dir image/file/or/folder \
  --dataset_json_path dataset.json \
  --model_path path/to/model \
  --model_config_path path/to/config
```

该命令会使用默认参数进行切片推理，并启动 FiftyOne App 进行交互式探索。

### 其他参数

支持 [`sahi predict`](#predict-命令用法) 命令的所有参数。

---

## `coco fiftyone` 命令用法

使用 FiftyOne UI 在 COCO 数据集上可视化和比较多个检测结果。

### 基本用法

你需要先将预测结果转换为
[COCO result JSON 格式](https://cocodataset.org/#format-results)。可以使用
[`sahi predict`](#predict-命令用法) 来生成该格式。

```bash
sahi coco fiftyone --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json cocoresult2.json
```

该命令会打开 FiftyOne 应用，可视化数据集并按误检排序比较两个检测结果。

### 选项

- `--iou_threshold 0.5` - 设置用于 FP/TP 分类的 IOU 阈值

---

## `coco slice` 命令用法

将大图像及其 COCO 格式标注切片为更小的块。

### 基本用法

```bash
sahi coco slice --image_dir dir/to/images \
  --dataset_json_path dataset.json
```

对图像和 COCO 标注进行切片，并导出到输出文件夹。

### 参数

**切片尺寸：**

- `--slice_size 512` - 设置切片的高度和宽度（默认：512）

**重叠率：**

- `--overlap_ratio 0.2` - 设置高度/宽度的重叠比例（默认：0.2）

**过滤：**

- `--ignore_negative_samples` - 排除没有标注的图像

**输出：**

- `--out_dir output/folder` - 指定输出目录

---

## `coco yolo` 命令用法

将 COCO 格式数据集转换为 YOLO 格式，用于 Ultralytics 训练。

> **Windows 用户：** 请以**管理员身份**打开 Anaconda prompt 或 Windows
> CMD，以正确创建符号链接。

### 基本用法

```bash
sahi coco yolo --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  --train_split 0.9
```

将 COCO 数据集转换为 YOLO 格式，并导出到 `runs/coco2yolo/exp` 文件夹。

### 参数

- `--train_split 0.9` - 设置训练集划分比例（默认：0.9）
- `--out_dir output/folder` - 指定输出目录

---

## `coco evaluate` 命令用法

计算预测结果的 COCO 评估指标（mAP、mAR）。

### 基本用法

你需要先将预测结果转换为
[COCO result JSON 格式](https://cocodataset.org/#format-results)。可以使用
[`sahi predict`](#predict-命令用法) 来生成该格式。

```bash
sahi coco evaluate --dataset_json_path dataset.json \
  --result_json_path result.json
```

计算 COCO 评估指标并将结果导出到输出文件夹。

### 参数

**指标类型：**

- `--type bbox` - 评估边界框检测（默认）
- `--type mask` - 评估实例分割掩码

**评分选项：**

- `--classwise` - 除整体指标外，额外计算每个类别的分数

**检测数量限制：**

- `--proposal_nums "[10 100 500]"` - 设置每张图像的最大检测数（默认：[100, 300,
  1000]）

**IOU 阈值：**

- `--iou_thrs 0.5` - 指定 IOU 阈值（默认：0.50:0.95 和 0.5）

**输出：**

- `--out_dir output/folder` - 指定输出目录

---

## `coco analyse` 命令用法

生成 COCO 预测的详细误差分析图。

### 基本用法

你需要先将预测结果转换为
[COCO result JSON 格式](https://cocodataset.org/#format-results)。可以使用
[`sahi predict`](#predict-命令用法) 来生成该格式。

```bash
sahi coco analyse --dataset_json_path dataset.json \
  --result_json_path result.json \
  --out_dir output/directory
```

生成全面的误差分析图并导出到指定文件夹。

### 参数

**分析类型：**

- `--type bbox` - 分析边界框检测（默认）
- `--type segm` - 分析实例分割掩码

**附加图表：**

- `--extraplots` - 生成额外的 mAP 柱状图和标注面积统计

**面积区间：**

- `--areas "[1024 9216 10000000000]"` - 定义分析的面积区间（默认：COCO 的小/中/大面积）

---

## `env` 命令用法

显示与 SAHI 相关的已安装包版本。

### 用法

```bash
sahi env
```

### 输出示例

```text
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torch version 2.1.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torchvision version 0.16.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   ultralytics version 8.3.86 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   transformers version 4.49.0 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   timm version 0.9.1 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   fiftyone version 0.14.2 is available.
```

---

## `version` 命令用法

显示当前安装的 SAHI 版本。

### 用法

```bash
sahi version
0.11.22
```

---

## 自定义脚本

所有脚本都可以从 [scripts 目录](https://github.com/obss/sahi/tree/main/scripts)
下载，并根据你的需求进行修改。

通过 pip 安装 SAHI 后，所有脚本都可以从任意目录调用：

```bash
python script_name.py
```

---

## 更多资源

- [预测工具](predict.md) -- 预测参数和可视化的 Python API
- [COCO 工具](coco.md) -- COCO 数据集操作的 Python API
- [模型集成](guides/models.md) -- 各框架的模型集成指南
- [交互式 Notebooks](notebooks.md) -- 所有框架的动手实践示例
