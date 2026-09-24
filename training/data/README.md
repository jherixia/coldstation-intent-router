# 冷站意图分类候选数据

- `train_reviewed.csv`包含180条候选训练问法。
- `validation_reviewed.csv`包含36条候选验证问法。
- 数据覆盖当前Demo的12种意图及处理类型。
- 数据已完成候选审核，仍待业务人员最终确认，不能标记为正式训练数据。
- 现有100条诊断回归数据已经用于问题定位，不是新的独立测试集。

字段：

- `text`：中文用户问法。
- `expected_intent`：监督训练或离线评测使用的期望标签。
- `label_basis`：人工标注依据。

`expected_intent`不得作为运行时分类输入。

文件SHA-256：

- `train_reviewed.csv`：`cac47c2f5dbeed93a9f5ff7448bbd25795c86368e987742a4143bc1e9ea947ae`
- `validation_reviewed.csv`：`a0ce90ea5a8629ed77bd017c12cd37e5258779590b46e899dfdb32cd0f222574`
