# 固定语义修复与验证结果

统一验证运行：`fixed-repair-validation-v2`。8 个任务 × 3 个开发初始状态，24/24 成功。所有动作通过原环境 step 执行，采用官方成功判定，预算仍为每回合 600 步。

**API 调用 0 次，输入/输出 token 均为 0；实验进程禁用网络连接。**

这是开发集上的修复验证，不是独立正式测试集成绩，也不能据此声称所有初始状态均能成功。录像沿用原协议，在第一次官方成功时停止；不额外追加松手或等待动作。

| 任务 | 修复后成功次数 | 状态 0 / 1 / 2 步数 | 成功录像 |
|---|---:|---|---|
| 1：奶油奶酪和黄油放入篮子 | 3/3 | 452 / 574 / 296 | [状态 0](../../runs/fixed-repair-validation-v2/task01_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task01_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task01_state002/video.mp4) |
| 2：打开炉灶并放上摩卡壶 | 3/3 | 230 / 236 / 233 | [状态 0](../../runs/fixed-repair-validation-v2/task02_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task02_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task02_state002/video.mp4) |
| 3：碗放入底层抽屉并关闭 | 3/3 | 333 / 327 / 348 | [状态 0](../../runs/fixed-repair-validation-v2/task03_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task03_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task03_state002/video.mp4) |
| 4：两个杯子分别放到左右盘子 | 3/3 | 272 / 261 / 263 | [状态 0](../../runs/fixed-repair-validation-v2/task04_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task04_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task04_state002/video.mp4) |
| 5：书放入收纳盒后槽 | 3/3 | 133 / 132 / 134 | [状态 0](../../runs/fixed-repair-validation-v2/task05_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task05_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task05_state002/video.mp4) |
| 6：杯子放盘上、布丁放盘子右侧 | 3/3 | 254 / 257 / 263 | [状态 0](../../runs/fixed-repair-validation-v2/task06_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task06_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task06_state002/video.mp4) |
| 8：两个摩卡壶放上炉灶 | 3/3 | 297 / 302 / 301 | [状态 0](../../runs/fixed-repair-validation-v2/task08_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task08_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task08_state002/video.mp4) |
| 9：杯子放入微波炉并关门 | 3/3 | 352 / 361 / 353 | [状态 0](../../runs/fixed-repair-validation-v2/task09_state000/video.mp4) · [状态 1](../../runs/fixed-repair-validation-v2/task09_state001/video.mp4) · [状态 2](../../runs/fixed-repair-validation-v2/task09_state002/video.mp4) |

## 排查及修复

- 任务 1：清除邻近高物体；修正薄盒夹持方向与下降深度。
- 任务 2：抓住旋钮凸起拨片；摩卡壶改抓顶部小柄。
- 任务 3：抓碗沿；按抽屉轴选把手；倾斜推关避开柜体和酒架。
- 任务 4：修正夹持点坐标系；以杯沿夹取代替抓整个杯身。
- 任务 5：按书本实际几何长轴对齐收纳槽。
- 任务 6：杯沿抓取与正确放置高度。
- 任务 8：提高搬运净空；两个壶使用分开的落点。
- 任务 9：抓杯柄；炉门外降高、正面送入；修正关门姿态和接触滞后。

详细迭代日志见 `docs/FIXED_REPAIR_LOG.md`；全部成功及失败尝试见 `docs/fixed_repair_attempts.jsonl`。每个回合目录包含结果、技能日志、逐步动作/状态、场景快照与双视角录像。

所有 24 个 MP4 均已实际解码检查：512×256 双视角，帧数与动作记录一致，末帧对应官方成功。

统一源码哈希：`29ccbc6c9f3b91351ce83c7fb0e225cd4dc394c8c74dde093df5aea80885e803`。源码归档：`runs/fixed-repair-validation-v2/source.zip`。

之前付费正式运行仍暂停；本轮修改后不能与旧版本成绩混合。

任务 1 状态 1 的 574 步成功轨迹已独立无 API 回放，再次成功，模拟状态最大差异为 0。回放记录：`reports/fixed_repair/replay_task01_state001/replay_result.json`。
