# Retrieved Data Analysis Recommendations

- 总样本数：**34429**
- 命中的 prototype 数：**70**
- 规范化重复样本数：**0**
- malformed 比例：**0.00%**
- prototype_score 中位数：**0.4739**
- prototype_score P10 / P90：**0.4538 / 0.5242**

## 过滤建议

- 先做 **normalized dedupe**，去除模板化和近重复窗口。
- 对 `prototype_score` 做全局下限，起点可先试 **0.50 ~ 0.65**。
- 再做 `per_prototype_min_quantile=0.1`，去掉每个 prototype 内部分数最低的 10%。
- 若头部 prototype 过多，可加 `per_prototype_topk` 控制单类上限。
- 对 `last_assistant_chars` 太短的样本直接丢弃，起点可先试 **20**。
- 对超长样本做上限，避免把冗长百科型回答全吃进去。