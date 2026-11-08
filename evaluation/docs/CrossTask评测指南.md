# CrossTask 下游微调与评测指南

本文对应本仓库已经跑通的 TR、SR、SF 全量流程：输入预提取视频特征和预训练 Adapter checkpoint，训练 Transformer 分类头，每个 epoch 在 test 划分上评测，记录所有 epoch 的最佳 top-1 accuracy。无需运行预训练。

## 1. 评测范围与协议

| 名称 | 程序任务名 | 输入与预测目标 | 类别数 |
|---|---|---|---|
| TR / Task Recognition | `task_cls` | 全视频特征 → 任务类别 | 83 |
| SR / Step Recognition | `step_cls` | 标注步骤时间区间内的特征 → 步骤类别 | 105 |
| SF / Step Forecasting | `step_forecasting` | 目标步骤之前的历史步骤特征 → 目标步骤类别 | 105 |

模型路径为 `S3D 预提取特征 → 预训练 MLP Adapter → Transformer 分类头`。引擎把 Adapter 设为 eval，并在 `torch.no_grad()` 内使用它；优化器只更新 Transformer 头。因此本文的“微调”具体指固定 Adapter、训练下游头，不是更新全部模型参数。

引擎入口：`evaluation/runtime/engines/main_train_taskhead.py`。每轮训练结束后调用测试函数；只有准确率严格提高时才保存最佳头。连续 50 轮未提高则早停，默认上限 1000 轮。

**指标解释：**当前协议用 test 准确率选择 epoch，并报告所有 epoch 中的最大值，未另外使用独立 validation 划分。它可用于复现当前代码的 best result；如果要做独立测试集上的无偏泛化评估，需要另设 validation 进行选模。原始 TR 划分还包含 33 个训练/测试重叠视频，本次保留了该源划分，因此结果不代表视频严格互斥协议。

## 2. 环境要求

原实验已跑通环境：macOS、Python 3.9.6、CPU、PyTorch 2.2.2、NumPy 1.26.4、PyYAML 6.0.3。CUDA 不是必需条件；Linux 可用 CUDA 版 PyTorch 加速，但该硬件组合未在本次运行验证。当前入口没有自动选用 Apple MPS。

主要依赖还包括 pandas、scikit-learn、scipy、matplotlib、tqdm、wandb；虽然默认关闭 wandb 记录，引擎仍在模块顶层 import wandb，因此需要安装。

安装环境后可检查依赖：

```bash
cd /path/to/TSS-unfolding
.venv-eval/bin/python -c 'import torch, numpy, yaml; print(torch.__version__, numpy.__version__, torch.cuda.is_available())'
```

新建与本次相同 Python 版本的环境：

```bash
python3.9 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r evaluation/requirements.lock.txt
export PYTHON="$PWD/.venv-eval/bin/python"
```

`evaluation/requirements.lock.txt` 是本次环境的完整版本快照。其他操作系统或 Python 版本可能需要调整包版本；CUDA 环境须安装与本机驱动兼容的 PyTorch 构建，不能把 macOS CPU 环境视为 CUDA 环境验证。仓库根目录 requirements.txt 面向预训练流程；下游评测推荐 evaluation/requirements.lock.txt。

CPU 内存随 batch 中最长视频序列和 batch size 增长。默认 batch size 16，内存不足可降到 8 或 4。数据包不含原视频；五个完整 checkpoint 合计约 7.5 GiB，还需要为解压数据、下游头和日志留空间。当前 CPU 已跑通，但没有测量峰值内存，因此不提供未经验证的最低内存/显存承诺。

## 3. 数据准备与路径

若已获得原数据压缩包，将其放到 `evaluation/CrossTask_required_full.tar.gz`。原工作区有综合校验记录 download_verification.json（不随代码上传）。原始下载校验覆盖 6,718 个压缩包成员。数据和权重未包含在 Git 仓库中，需自行放置；下载记录描述原运行工作区，不提供公开下载链接。数据包包含本流程依赖的特征、时间戳、标注和划分元数据，不含原视频。

尚未解压时执行：

```bash
mkdir -p evaluation/data
tar -xzf evaluation/CrossTask_required_full.tar.gz -C evaluation/data
```

目标目录结构：

```text
evaluation/data/CrossTask/
├── crosstask_release/annotations/<task_id>_<video_id>.csv
└── feats_3segs_no_mask/
    ├── <video_id>/video.npy
    ├── <video_id>/segment_time.npy
    ├── task_cls_train_split.pickle
    ├── task_cls_test_split.pickle
    ├── step_cls_train_split.pickle
    ├── step_cls_test_split.pickle
    ├── step_cls_task2step_meta.pickle
    ├── step_cls_set_of_steps.pickle
    ├── step_forecasting_train_split.pickle
    ├── step_forecasting_test_split.pickle
    ├── step_forecasting_task2step_meta.pickle
    └── step_forecasting_set_of_steps.pickle
```

`video.npy` 的数据约定是 `(片段数, 3, 512)`；识别任务沿中间维度取均值，得到 512 维片段特征。`segment_time.npy` 对齐片段内子片段的时间戳，SR 和 SF 用它与标注时间对齐。CSV 每行是步骤 ID、开始秒数、结束秒数；ID 按所属任务的步骤元数据映射到全局分类标签。

TR 的 split pickle 是任务到样本列表的映射，样本采用 `task_id***video_id`。SR/SF 的 split pickle 是标注文件列表；加载器取文件名并在配置的本地 annotations 目录中寻找文件，因此列表内的远程绝对路径不必逐项替换。

| YAML 字段 | 用途 |
|---|---|
| `cross_dir`、`cross_task_annot_dir` | CrossTask 根目录 |
| `cross_task_s3d_feat_dir` | 特征及 split/meta pickle 所在目录 |
| `cross_task_annoataion_dir` | CSV 标注目录；**保留代码中的这个拼写** |
| `cross_task_video_dir` | 原视频路径字段；当前三任务读取预提取特征，不需要原视频 |

一键脚本的 `--data-dir` 指向上面 CrossTask 根目录，会自动配置这些路径。默认全量实际加载样本数：TR 1819 / 483，SR 13746 / 3448，SF 10780 / 2703（train / test）。这些是数据集样本数，不能等同于去重视频数。

### SF 的历史过滤

`forecast_history_only: true` 是本次已跑通协议的一部分。`coin_step_forecasting_history: 1` 虽然字段名带 COIN，CrossTask SF 也使用它，表示使用目标之前的 1 个标注步骤。只选择目标开始前已经结束的历史子片段，并剔除没有有效历史的样本；因此 SF 的样本数小于直接按所有标注步骤计数的结果。该逻辑位于 `evaluation/runtime/datasets/cross_task.py`，迁移代码时需要一起保留。

### 只用部分数据跑通流程

可以缩小训练和测试 split，但必须同时保留被引用的视频特征、SR/SF 时间戳和 CSV。建议保留完整标签元数据和原有类别数，避免改动标签映射。只删数据文件、不改 split 会导致 FileNotFoundError。测试子集也要有有效样本，SF 要有有效历史。子集结果仅用于检查流程，不能与全量结果直接比较。

## 4. 配置 checkpoint

五个下载文件均在 `evaluation/checkpoints/`：

```text
task.pth
task-step.pth
task-step-state.pth
task-step-state-step.pth
task-step-state-step-task.pth
```

参考实验三任务全部使用 **task-step-state-step-task.pth**；其他四个未在本次全量实验中进行微调对比。它们可以通过一键脚本的 `--checkpoint` 分别指定，脚本每次只使用一个 checkpoint。

加载器读取 `torch.load(path)['state_dict']`，经 `trim()` 处理键名后，用 `strict=True` 加载到 Adapter 内部网络。其他阶段的预测头和 optimizer 状态不会由下游训练入口恢复。文件即使含完整优化器状态，也不表示会继续预训练。

Adapter 的结构从 `pretrain_config` 指定的 YAML 构建，默认 `evaluation/runtime/configs/CoP_pretrain/CoP_pretrain_by_stage.yml`：`adapter_mlp`，输入 512、瓶颈 128、输出 512。更换 checkpoint 时必须与该结构匹配；不要用 `strict=False` 掩盖不匹配。该 YAML 中的远程预训练数据路径在此入口不用于启动预训练，无需准备 HowTo100M、wikiHow 或 COIN 数据。

下游最佳头 `TaskHead-*_model_best.pth` 与预训练 Adapter checkpoint 是不同文件：前者保存训练后的 Transformer，不能直接当成 `--checkpoint` 参数传入当前训练入口。部署或独立重评应同时提供原 Adapter、最佳头、相同模型配置及标签映射；当前 quick start 执行训练加逐轮评测，不提供单独加载最佳头的 eval-only 模式。

## 5. Quick start：一键运行

从仓库根目录执行，默认依次运行三个任务，固定 CPU、关闭 wandb：

```bash
bash evaluation/quick_start.sh
```

脚本优先使用环境变量 `PYTHON` 指定的解释器，其次使用 `.venv-eval/bin/python`，否则使用 `python3`。新环境需先按第 2 节设置 `PYTHON`。

自定义数据和 checkpoint：

```bash
bash evaluation/quick_start.sh \
  --data-dir /data/CrossTask \
  --checkpoint /models/task-step-state-step-task.pth
```

只运行 SR，或在 CUDA 设备上运行：

```bash
bash evaluation/quick_start.sh --tasks step_cls
bash evaluation/quick_start.sh --device cuda:0 --tasks task_cls step_cls step_forecasting
```

用 1 个 epoch 检查整个训练与评测入口：

```bash
bash evaluation/quick_start.sh --epochs 1 --patience 1 --batch-size 4
```

这个命令仍遍历所选任务的完整数据；减少 epoch 不等于缩小数据。它的结果不能作为收敛后的最佳性能。

仅生成配置并做基本文件检查：

```bash
bash evaluation/quick_start.sh --prepare-only
```

此模式检查 checkpoint 非空、split/meta 文件和基本目录存在，不检查每个样本或 checkpoint 权重兼容性；后两项由真正运行时的数据读取与 strict 加载发现。

每次执行默认创建唯一的 `evaluation/experiments/<时间戳>/`，不会覆盖原始实验。指定 `--output-dir` 时须使用一个尚不存在的目录。生成的 YAML 中路径均转为本机绝对路径，避免配置模板绑定某一台机器。可调整 batch size、epochs、patience、seed、num-workers；更多参数见 `--help`。需修改学习率或 Transformer 结构时，可先生成配置，编辑后按第 7 节直接调用引擎。

脚本遇到一个任务失败会停止，保留该任务 console 日志；重新启动会从头训练，不自动断点恢复。默认顺序执行以减少资源争用。

## 6. 关键配置参数

三个模板位于 `evaluation/configs/{task_cls,step_cls,step_forecasting}_Full.yml`。一键脚本据此生成本机配置。

| 参数 | 本次值 | 说明 |
|---|---|---|
| `pretrain_config` | CoP_pretrain_by_stage.yml | 构建 Adapter 的结构配置，不启动预训练 |
| `model_*_head_name` | downstream_transformer | 三种任务均选择 Transformer 下游头 |
| `model_video_pretrained_dim` | 512 | 输入特征维度，与数据和 Adapter 对齐 |
| `model_*_segment_hidden_dim` | 512 | Transformer 隐藏维度 |
| `model_*_tx_nhead` | 8 | 注意力头数；隐藏维度须能被头数整除 |
| `model_*_tx_dim_feedforward` | 1024 | 前馈层维度 |
| `model_*_tx_dropout` | 0 | dropout 概率 |
| `model_*_tx_activation` | relu | 激活函数 |
| `model_*_time_pos_embed_type` | absolute_learned_1D | 时间位置编码 |
| `model_*_max_time_ids_embed` | 500 | 时间位置嵌入容量，需与输入序列及实现兼容 |
| `model_*_classifier_hidden_dim` | TR 128；SR/SF 768 | 分类器隐藏维度 |
| `model_*_num_classes` | TR 83；SR/SF 105 | 分类输出维度，与标签映射一致 |
| `task_head_batch_size` | 16 | 变长序列会按 batch 最长样本补齐 |
| `task_head_optimizer` | adam | 当前入口实现的优化器 |
| `task_head_learning_rate` | 1e-4 | 下游头学习率 |
| `task_head_weight_decay` | 1e-3 | 权重衰减 |
| `task_head_num_epochs` | 1000 | 最大训练轮数 |
| `task_head_early_stop_patience` | 50 | 连续多少轮无严格提升后停止 |
| `task_head_lr_warm_up` | false | false 时本次没有启用 warmup/cosine 调度 |
| `task_head_warmup_steps` | 5000 | 仅启用 warmup 时生效 |
| `always_save_best_task_head_each_epoch` | true | 发现更好结果时覆盖最佳头文件 |
| `seed` | 42 | NumPy、random、torch 随机种子；不能保证跨平台完全一致 |
| `num_workers` | 0 | 主进程加载；避开本次遇到的 spawn/共享内存问题 |
| `device` | 本次实际 cpu | 原模板虽写 cuda:0，无 CUDA 时入口会回退 CPU |
| `forecast_history_only` | true | SF 使用严格的已观测历史 |
| `coin_step_forecasting_history` | 1 | SF 使用的历史步骤数 |
| `final_acc_output` | CSV 路径 | 每个任务完成后追加最佳结果 |
| `checkpoint_dir`、`log_dir` | 输出目录 | 下游最佳头和逐轮日志 |

`*` 分别替换成 `task_cls`、`step_cls`、`step_forecasting`。数据维度、类别数和 Adapter 结构不应只为消除报错随意修改。布尔参数建议修改 YAML；当前参数解析器使用 `type(bool)`，命令行字符串 `False` 不可靠。

## 7. 直接调用引擎

推荐使用一键脚本切换权重和生成本机路径。如需运行编辑后的配置，先进入 evaluation/runtime 目录（其导入依赖当前工作目录）：

```bash
cd evaluation/runtime
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../.venv-eval/bin/python engines/main_train_taskhead.py \
  --cfg /absolute/path/to/step_cls.yml \
  --checkpoint /absolute/path/to/task-step-state-step-task.pth \
  --downstream_dataset_name CrossTask \
  --downstream_task_name step_cls \
  --notes CustomRun --use_wandb 0
```

请使用直接脚本入口，不用 `runpy` 包装后开启多进程 DataLoader。本次曾遇到的 macOS spawn 路径错误和 OpenMP SHM 错误已通过直接子进程、`num_workers: 0`、线程数 1 绕开。

## 8. 监控与读取结果

一键脚本启动时打印输出目录，运行期间控制台完整输出写入 `<task>.console.log`，引擎日志写入 `logs/`。例如：

```bash
tail -f evaluation/experiments/<时间戳>/step_cls.console.log
rg 'Best acc so far|Finished testing|Early stopping|Finished training and testing' evaluation/experiments/<时间戳>/logs
```

需要每 10 分钟查看时，可在另一个终端使用：

```bash
while true; do
  date
  rg 'Best acc so far' evaluation/experiments/<时间戳>/logs | tail -3
  sleep 600
done
```

该循环只打印日志，不提供桌面推送，结束后按 Ctrl-C 停止。日志停止更新本身不能说明训练成功；需查找结束标记并确认脚本退出状态为 0。

完成后的输出：

| 文件 | 内容 |
|---|---|
| `summary.csv` | 新脚本汇总的任务名、最佳准确率百分数、最佳 epoch、日志路径，带表头 |
| `raw_results.csv` | 原引擎追加的完整精度结果，无表头 |
| `<task>.yml` | 本次实际配置 |
| `logs/*withAdapter(<task>)*.log` | 每轮训练测试、最佳值、完成标记 |
| `checkpoints/<运行目录>/TaskHead-<task>_model_best.pth` | 最佳 Transformer 头，含 cfg、epoch、state_dict、optimizer |

原引擎 raw CSV 列顺序为 `curr_time, finish_time, logger, ckp_path, ckp_idx, ds_dataset, ds_taskname, model_structure, best_acc, best_epoch`。历史代码将 finish_time 标成 Z，但使用本机时间，不能据此认定为 UTC。summary 使用日志中保留两位小数的值，raw CSV 保留更高精度。

## 9. 本次已完成的参考结果

使用 task-step-state-step-task.pth、Transformer、seed 42、batch 16、CPU，保留源数据划分及 SF 历史过滤：

| 任务 | 最佳 test top-1 accuracy | 最佳 epoch | 停止 epoch |
|---|---:|---:|---:|
| TR | 89.65% | 6 | 56 |
| SR | 57.19% | 16 | 66 |
| SF | 39.07% | 6 | 56 |

原始运行来自 VLP_procedure 工作区；本仓库在 `evaluation/reference_results.csv` 保存去除本机路径后的参考数值，不包含原训练日志、数据及模型权重。上述数值属于本仓库当前协议，解释时需同时保留第 1 节的数据重叠和 test 选模说明。

## 10. 常见问题

- **缺文件**：确认 `--data-dir` 指向 CrossTask 根目录，核查 split 中所有视频的 `video.npy`；SR/SF 还要有 `segment_time.npy` 和 CSV。
- **strict 加载报 missing/unexpected keys 或 size mismatch**：确认输入的是预训练 Adapter checkpoint，并核对 pretrain_config 的 Adapter 结构。不要混入下游 TaskHead 文件。
- **CUDA 请求失败**：新脚本会明确报错；改用 `--device cpu`，或安装可用的 CUDA PyTorch。
- **CPU 跑得慢或内存不足**：降低 batch size；也可减少 epoch 做流程检查，正式性能实验保留足够轮数和早停。
- **SF 初始化暂时没有 batch 日志**：它会先检查所有候选样本的历史有效性，完成后打印保留/剔除数量。
- **同一参数重跑结果略有差异**：变长样本顺序、库版本、CPU/CUDA 数值和步骤集合的迭代次序均可能影响结果。`seed: 42` 不等于跨环境逐位复现；独立加载头时应保留训练时标签映射。
- **只想评测已训练好的头**：当前入口会初始化并训练新头，不能用 `--epochs 0` 替代独立评测，也不能把 best head 传成 Adapter checkpoint。


## 11. 本机路径配置与公开提交

仓库只保留通用相对路径和示例路径，不包含开发者的用户名、主目录或主机名。将下面示例替换为你自己的实际位置：

```bash
# 在仓库根目录执行；路径带空格时保留引号。
export PYTHON="/path/to/python"
bash evaluation/quick_start.sh \
  --data-dir "/path/to/CrossTask" \
  --checkpoint "/path/to/pretrained-adapter.pth" \
  --output-dir "/path/to/new-experiment-directory"
```

- `PYTHON`：已安装 evaluation/requirements.lock.txt 依赖的 Python 解释器。若使用仓库内 `.venv-eval`，无需设置。
- `--data-dir`：CrossTask 根目录，内部结构按第 3 节放置。省略时为 evaluation/data/CrossTask。
- `--checkpoint`：预训练 Adapter 文件。省略时为 evaluation/checkpoints/task-step-state-step-task.pth。
- `--output-dir`：尚不存在的输出目录。省略时自动生成 evaluation/experiments 下的唯一目录。

脚本把这些值写入运行时配置：特征路径映射到 cross_task_s3d_feat_dir，标注路径映射到 cross_task_annoataion_dir，输出路径映射到 log_dir、checkpoint_dir、final_acc_output。ExpLogger 辅助工具也使用配置的 log_dir，不依赖固定用户目录。

运行生成的 YAML、日志、CSV、模型 checkpoint 可能包含实际绝对路径或配置对象，属于本机运行产物。默认 experiments、data、checkpoints 和虚拟环境均已在 .gitignore 中排除；若指定仓库内其他输出目录，应自行将该目录加入 .gitignore。公开分享日志或结果前，需将实际路径替换成 `<DATA_DIR>`、`<CHECKPOINT>`、`<OUTPUT_DIR>` 等占位符。训练入口不再主动记录主机名。

Git 作者信息属于提交元数据，与代码中的路径配置独立。提交前可用 `git log -1 --format=fuller` 检查身份，按个人发布需求配置仓库级 user.name 和 user.email。修改当前文件不会删除已有 Git 历史中的旧内容。
