# 冷站能效分层意图识别 Demo

独立的中文意图识别实验项目，执行顺序为：

`L0 规则 → L1 BGE → L2 Qwen → L3 澄清 → 模拟业务响应`

项目只返回模拟说明，不连接公司Agent、业务数据库或设备控制接口。

## 支持的意图与处理类型

- `mon.read`
- `mon.compare`
- `opt.read`
- `nav.opt`
- `nav.overview`
- `nav.sim`
- `nav.history`
- `clarify`
- `reject_control`
- `internal.explain`
- `internal.savings_estimate`
- `internal.capability_missing`

其中三个`internal.*`名称仅用于本Demo内部处理，不表示公司现网正式意图代码。

## 环境准备

建议使用Python 3.11或3.12。RTX 5070主机应先根据其驱动和CUDA环境安装合适的GPU版PyTorch，再安装本项目依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 先按 https://pytorch.org/get-started/locally/ 安装GPU版PyTorch
python -m pip install -r requirements.txt
```

L1使用`BAAI/bge-small-zh-v1.5`，代码设置了`local_files_only=True`，不会在运行时自动下载模型。请预先把模型放入Hugging Face缓存，或设置本地路径：

```powershell
$env:BGE_MODEL_PATH = "D:\models\bge-small-zh-v1.5"
```

L2默认连接本机Ollama：

- API：`http://127.0.0.1:11434/v1/chat/completions`
- 模型：`qwen2.5:1.5b`

在目标主机自行安装Ollama并准备模型；模型权重不保存在Git中。配置可通过环境变量覆盖：

```powershell
$env:L2_API_URL = "http://127.0.0.1:11434/v1/chat/completions"
$env:L2_MODEL = "qwen2.5:1.5b"
$env:L2_TIMEOUT_SECONDS = "30"
# 仅在接口确实需要认证时，在本机设置；不要提交真实密钥
$env:L2_API_KEY = ""
```

## 运行

交互式Demo：

```powershell
.\run.ps1
```

也可指定Python解释器：

```powershell
.\run.ps1 -PythonPath "D:\Python312\python.exe"
```

基础24条评测：

```powershell
.\run.ps1 -Evaluate
```

其他固定评测：

```powershell
python evaluate_independent.py
python evaluate_business_scope.py
python evaluate_coldstation_holdout.py
```

评测脚本的输入和输出路径均相对于脚本目录。生成的结果、汇总和报告由`.gitignore`排除。

## 训练准备数据

审核后的候选数据位于`training/data/`：

- `train_reviewed.csv`：180条
- `validation_reviewed.csv`：36条

这些数据覆盖12种意图及处理类型，已完成候选审核，但仍待业务人员最终确认，不能标记为正式训练集。现有100条诊断回归数据也不是新的独立测试集。

## 固定配置

当前L1实验阈值保持为：

```text
MIN_SIMILARITY = 0.80
MIN_MARGIN = 0.06
```

本次Git迁移未修改L0-L3分类规则、原型、阈值或Qwen提示词。
