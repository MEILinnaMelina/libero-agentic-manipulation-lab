# GPT-6 多状态运行结果

运行 `gpt6-all10-states456-20260908-v1`。任务 0, 1, 2, 3, 4, 5, 6, 7, 8, 9；官方初始状态 4, 5, 6，每个预定任务与状态组合执行一次。

完成 **30/30** 回合，成功 **27**。这是一批限定状态的检查，不是完整 470 回合正式评测，不与旧版本结果混合。

共 **212** 次 GPT-6 Astra 调用，费用按已记录用量估算 **$8.2216**，不是账单。API 错误 0 回合，未知用量请求 0 条。

| 任务 | 状态 4 | 状态 5 | 状态 6 | 成功次数 | API 次数 | 估计美元 |
|---|---|---|---|---:|---:|---:|
| 0 汤罐和番茄酱放篮子 | [成功 · 392 步](../../runs/gpt6-all10-states456-20260908-v1/task00_state004/video.mp4) | [成功 · 385 步](../../runs/gpt6-all10-states456-20260908-v1/task00_state005/video.mp4) | [成功 · 288 步](../../runs/gpt6-all10-states456-20260908-v1/task00_state006/video.mp4) | 3/3 | 26 | 1.2331 |
| 1 奶酪和黄油放篮子 | [成功 · 428 步](../../runs/gpt6-all10-states456-20260908-v1/task01_state004/video.mp4) | [成功 · 430 步](../../runs/gpt6-all10-states456-20260908-v1/task01_state005/video.mp4) | [失败 · 541 步](../../runs/gpt6-all10-states456-20260908-v1/task01_state006/video.mp4) | 2/3 | 26 | 1.1991 |
| 2 打开炉灶并放摩卡壶 | [成功 · 228 步](../../runs/gpt6-all10-states456-20260908-v1/task02_state004/video.mp4) | [成功 · 229 步](../../runs/gpt6-all10-states456-20260908-v1/task02_state005/video.mp4) | [成功 · 233 步](../../runs/gpt6-all10-states456-20260908-v1/task02_state006/video.mp4) | 3/3 | 15 | 0.4557 |
| 3 碗放抽屉并关闭 | [成功 · 295 步](../../runs/gpt6-all10-states456-20260908-v1/task03_state004/video.mp4) | [失败 · 339 步](../../runs/gpt6-all10-states456-20260908-v1/task03_state005/video.mp4) | [成功 · 276 步](../../runs/gpt6-all10-states456-20260908-v1/task03_state006/video.mp4) | 2/3 | 15 | 0.6154 |
| 4 两个杯子放左右盘子 | [成功 · 277 步](../../runs/gpt6-all10-states456-20260908-v1/task04_state004/video.mp4) | [成功 · 275 步](../../runs/gpt6-all10-states456-20260908-v1/task04_state005/video.mp4) | [成功 · 270 步](../../runs/gpt6-all10-states456-20260908-v1/task04_state006/video.mp4) | 3/3 | 24 | 0.8800 |
| 5 书放入收纳盒后槽 | [成功 · 132 步](../../runs/gpt6-all10-states456-20260908-v1/task05_state004/video.mp4) | [成功 · 143 步](../../runs/gpt6-all10-states456-20260908-v1/task05_state005/video.mp4) | [成功 · 138 步](../../runs/gpt6-all10-states456-20260908-v1/task05_state006/video.mp4) | 3/3 | 12 | 0.4087 |
| 6 杯子放盘上、布丁放右侧 | [成功 · 255 步](../../runs/gpt6-all10-states456-20260908-v1/task06_state004/video.mp4) | [成功 · 257 步](../../runs/gpt6-all10-states456-20260908-v1/task06_state005/video.mp4) | [成功 · 252 步](../../runs/gpt6-all10-states456-20260908-v1/task06_state006/video.mp4) | 3/3 | 24 | 0.8443 |
| 7 汤罐和奶酪放篮子 | [成功 · 314 步](../../runs/gpt6-all10-states456-20260908-v1/task07_state004/video.mp4) | [成功 · 329 步](../../runs/gpt6-all10-states456-20260908-v1/task07_state005/video.mp4) | [成功 · 383 步](../../runs/gpt6-all10-states456-20260908-v1/task07_state006/video.mp4) | 3/3 | 27 | 0.9827 |
| 8 两个摩卡壶放炉灶 | [成功 · 320 步](../../runs/gpt6-all10-states456-20260908-v1/task08_state004/video.mp4) | [成功 · 328 步](../../runs/gpt6-all10-states456-20260908-v1/task08_state005/video.mp4) | [成功 · 315 步](../../runs/gpt6-all10-states456-20260908-v1/task08_state006/video.mp4) | 3/3 | 24 | 0.8853 |
| 9 杯子放微波炉并关门 | [失败 · 600 步](../../runs/gpt6-all10-states456-20260908-v1/task09_state004/video.mp4) | [成功 · 281 步](../../runs/gpt6-all10-states456-20260908-v1/task09_state005/video.mp4) | [成功 · 355 步](../../runs/gpt6-all10-states456-20260908-v1/task09_state006/video.mp4) | 2/3 | 19 | 0.7174 |

## 失败与停止原因

- 任务 1，状态 6：replan_budget；技能失败：control_error, control_error, control_error, control_error。
- 任务 3，状态 5：replan_budget；技能失败：placement_unstable, control_error, control_error, control_error。
- 任务 9，状态 4：step_budget；技能失败：mechanism_contact, control_error, control_error。

整批停止原因：计划回合全部结束。

## 协议与录像核验

沿用官方成功判定、600 动作步、3 次重规划与首次成功终止。独立预算上限 $20。不改变物理、初始状态或成功条件，不在运行中调参，不重跑失败回合。

已解码核验 30 段双视角视频，512×256、20 fps；帧数与逐步轨迹一致。成功回合最终官方目标均为真。视频在首次成功时结束，不包含额外松手稳定性测试。

输入 589,647 token，输出 17,053 token；其中缓存输入 0、缓存写入 589,011、普通输入 636。计价单价随运行配置保存。

执行代码 SHA-256：`8c7738eecf1f8926ae7e89af6bd15366c777a5f68617c6f97ee99fc246a188dc`。完整执行源码与配置见运行目录 `source.zip`；每回合保留技能、请求响应、初末场景、逐步动作和状态。
