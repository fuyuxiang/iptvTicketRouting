# IPTV 客服工单自动分类与分流

面向运营商 IPTV 客服场景：对工单文本做自动分类（故障大类），并基于分类结果做自动分流（映射到分配部门/队列），降低人工初筛成本。

## 1. 环境要求
- Python 3.6+
- 依赖见 requirements.txt（scikit-learn 0.20.x、Flask 1.1.x、jieba）

## 2. 数据
默认读取：`data/iptv_masked.csv`

字段约定：
- 文本字段：故障描述 / 错误码 / 涉及频道/内容 / 业务类型 / 故障大类 / 故障小类（会拼接成模型输入）
- 标签字段：故障大类
- 分流字段：分配部门（用于统计“故障大类 -> 分配部门”的多数投票映射）

## 3. 训练模型
```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/train.py --config configs/config.yaml
```

训练输出：
- models/ticket_clf.joblib          # sklearn pipeline
- models/label_map.json             # 分类标签列表
- models/route_map.json             # 故障大类 -> 分配部门（多数投票）映射
- 终端打印离线评估（train/test split）

## 4. 启动在线服务
先确保 models/ 下有训练产物，再启动：

开发模式：
```bash
export PYTHONPATH=src
export IPTV_CONFIG=configs/config.yaml
python -m api.wsgi
```

生产模式（建议）：
```bash
export PYTHONPATH=src
export IPTV_CONFIG=configs/config.yaml
gunicorn -w 4 -b 0.0.0.0:8080 api.wsgi:app --access-logfile - --error-logfile -
```

## 5. API
- GET /healthz
- POST /predict

请求：
```json
{
  "ticket_id": "IPTV-xxx",
  "text": "机顶盒一直提示网络异常，无法播放"
}
```

响应（示例）：
```json
{
  "ticket_id": "IPTV-xxx",
  "label": "无信号",
  "confidence": 0.83,
  "dept": "区域装维团队",
  "queue": "QUEUE_FIELD",
  "auto_routed": true
}
```

说明：
- label = 预测故障大类
- dept = 基于 route_map（多数投票）得到的推荐分配部门
- queue = 可选二次映射（dept_to_queue）
- auto_routed = 置信度 >= threshold_auto_route 才会自动分流，否则转人工
