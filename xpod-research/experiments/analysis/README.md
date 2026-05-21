# xPod 论文图表生成模块

这个模块用于生成 xPod 论文的 6 张核心图表，对比 xPod 与 Random Baseline 在各项指标上的表现。

## 图表列表

1. **Fig 1**: 累计冷启动收敛曲线 - 展示数据局部性带来的收敛速度优势
2. **Fig 2**: 累计数据传输量 - 展示数据移动成本的 2x 降低
3. **Fig 3**: GPU contention CDF - 展示 vGPU 资源争用的分布情况
4. **Fig 4**: JCT 长尾对比 - 展示 Job Completion Time 的百分位对比
5. **Fig 5**: Sensitivity 趋势 (3 子图) - 展示对 GPU capacity、vGPU 数量和数据节点数的敏感度
6. **Fig 6**: 节点负载热力图 - 展示任务在各 GPU 节点上的时间分布

## 使用方法

### 生成全部图表

```bash
python -m experiments.analysis.make_paper_figures
```

### 只生成指定图表

```bash
python -m experiments.analysis.make_paper_figures --figures 1,4,5
```

### 自定义参数

```bash
python -m experiments.analysis.make_paper_figures \
    --xpod-csv /path/to/xpod.csv \
    --baseline-csv /path/to/baseline.csv \
    --sensitivity-dir /path/to/sensitivity/ \
    --trace-csv /path/to/trace.csv \
    --output-dir /path/to/output/ \
    --log-level DEBUG
```

## 数据来源

- **主实验数据**: 从主实验运行结果获得，默认路径 `results/raw/v1_final/`
  - `xpod_4vgpu_cap50.csv`: xPod 调度结果
  - `baseline_4vgpu_cap50.csv`: Random Baseline 调度结果

- **Sensitivity 实验数据**: 从 sensitivity 实验获得，默认路径 `results/raw/sensitivity/`
  - `A_gpucap{cap}_xpod.csv` / `A_gpucap{cap}_baseline.csv`: GPU capacity 敏感度
  - `B_vgpu{n}_xpod.csv` / `B_vgpu{n}_baseline.csv`: vGPU 数量敏感度
  - `C_data{n}_xpod.csv` / `C_data{n}_baseline.csv`: 数据节点数敏感度

- **原始 trace 数据**: 用于获取任务 duration，默认路径 `datasets/alibaba-cluster-trace-gpu-v2020/processed/scene4-joint/xpod_requests.csv`

## 输出位置

生成的图表会保存到 `results/figures/` 目录下（可通过 `--output-dir` 自定义）。

每张图同时生成两种格式：
- **PNG**: 300 DPI，用于演示文稿
- **PDF**: 矢量图，用于论文排版

文件命名格式：`fig{number}_{short_name}.{extension}`

## 如何添加新图

如需添加新图表，按以下步骤操作：

1. 在 `make_paper_figures.py` 中添加一个新的绘图函数：
```python
def make_fig7_new_figure(xp_df: pd.DataFrame, bs_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    # 你的绘图逻辑
    return fig
```

2. 在 `main()` 函数中添加生成逻辑：
```python
if 7 in figures_to_gen:
    logger.info('Generating Fig 7: New Figure')
    fig7 = make_fig7_new_figure(xp_df, bs_df)
    save_figure(fig7, 'fig7_new_figure', args.output_dir)
```

3. 在 `figures_to_gen` 列表中添加新的编号。

## 依赖

- pandas
- matplotlib
- numpy

已在项目根目录 `requirements.txt` 中声明。
