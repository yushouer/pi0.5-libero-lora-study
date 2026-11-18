# 如何将本实验包发布到 GitHub

这个文件夹是一份已经整理好的 **GitHub README + 结果图表 + 数据与失败分析**。它不包含你服务器上实际训练过的 openpi 源码差异、checkpoint 或视频，因此不能独立复现模型训练；发布前需要补充代码。

## 推荐方式：创建独立项目仓库

1. 创建一个新的项目仓库（例如 `pi05-libero-lora-study`）。
2. 将这个文件夹中的 `README.md`、`assets/`、`results/`、`reports/`、`scripts/plot_results.py` 和 `.gitignore` 拷贝到新仓库根目录。
3. 从服务器**记录并发布真实的源码改动**：

```bash
cd ~/xyj/openpi
# 保存版本信息，确保别人能重建上游环境
git rev-parse HEAD
git status --short
# 查看训练配置 / Episode 数据加载的实际变更
git diff -- src/openpi/training/config.py src/openpi/training/data_loader.py
```

4. 把修改后的配置与筛选代码以 **patch 文件或明确的源码模块** 公开；还应补齐你实际运行的 `examples/libero/eval_book.py`、`scripts/eval_book_val_loss.py` 及对照实验执行命令。**切勿上传服务器密码、Token、代理凭证、权重文件或其他人的数据**。
5. 核对 README 中相对路径：这个包使用 `results/train_episodes.json` 和 `results/val_episodes.json`，而原服务器记录路径为 `results/book_task09/...`。发布时选一种统一布局并相应更新命令。
6. 在 README 中记录你实际使用的 `git rev-parse HEAD`、`uv.lock` 或依赖版本；不要凭记忆填写。
7. 检查许可与引用：openpi、LIBERO 和 LeRobot 的代码、数据及权重具有各自许可，避免未经确认直接重新发布大型资产。

## 绘图文件

独立运行：

```bash
pip install matplotlib
python scripts/plot_results.py
```

所有曲线使用 `results/checkpoint_metrics.csv` 的真实结果；失败分类采用 `results/failure_cases.csv` 的 21 条人工视频标注。图中英文标签便于跨平台渲染，README 保留中文算法解释。

## 建议提交记录

```text
Add pi0.5 LoRA experiment README and evaluation figures
Document episode filtering, checkpoints and validation metrics
Add failure analysis for 21 LIBERO book-task rollouts
```
