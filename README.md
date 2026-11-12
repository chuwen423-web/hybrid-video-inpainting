# Hybrid Video Inpainting（Week 4 – Week 5 代码）

COMP-8977 Internship/Project-2 · University of Windsor · Fall 2026 · Team 3
Mingran Chen · Yarong Zhang · Zhihan Zhang
Supervisor: Jessica Chen · Course Instructor: Dr. Prashanth C. Ranga

这是视频里物体删除 / 替换原型的代码，所有程序都在本机运行，网页只监听 127.0.0.1。
- 用 YOLOX 检测物体，用 SAM 2 逐帧追踪得到 mask。
- 删除时用背景板或 LaMa 补洞，并做去闪烁、去光晕。
- 提供三个本地网页：演示界面、视频编辑器、多目标追踪器。

## 仓库内容

| 文件夹 | 内容 |
|---|---|
| `week5_multi_track/` | **多目标识别与追踪**（http://127.0.0.1:8768）。<br>YOLOX 逐帧检测，把检测框连成实例；SAM 2 同时追踪多个物体，并合并重复编号。<br>网页上有时间轴、物体改名和测试结果页。说明见 `README.md`，设计决定见 ADR-006 |
| `week4_video_editor/` | **视频编辑器**（http://127.0.0.1:8767）。<br>流程：上传 → 检测 → 选择物体或手动框选 → SAM 2 追踪 → 删除或替换 → 下载 |
| `week4_temporal/` | **去闪烁和去光晕**：<br>• `temporal.py`：稳定 mask、背景板、光流平滑、自动退回逐帧修补<br>• `demo_app.py`：方法对照演示界面<br>• `eval_h3.py`：全局模型结果评估<br>设计决定见 ADR-005 |
| `week3_video_pipeline/` | 被 Week 4/5 调用的第三周代码：<br>• `detect.py`：YOLOX 检测<br>• `pipeline.py`：关键帧、SAM 2、LaMa 裁剪修补、读写<br>• `evaluate_video.py`：闪烁和保真度指标<br>• `replace_book.py`：贴图对齐<br>另有 `config/clips.json` 和 ADR-004 |
| `week3_sdxl_inpaint/` | `sdxl_inpaint.py`：ComfyUI 客户端，被 `replace_book.py` 导入。只有用 SDXL 生成替换内容时才需要 ComfyUI |
| `week2_mask_tracking/` | `config/clips.json`：SAM 2 模型配置和权重路径。运行环境 `.venv` 也约定放在这个文件夹下 |
| `week1_image_validation/tools/` | 第一周的工具函数：mask 操作、接缝指标、合成、可视化 |
| `test_clips/` | 测试视频放这里（不上传） |

**没有上传的内容**：模型权重、测试视频、运行结果（`results/`、`work/`、`jobs/`）、实验记录，以及周报、中期报告等文档。

## 需要另外下载的模型

下载后放到下表的路径（相对仓库根目录）。文件名必须一致，因为代码按这些路径读取。

| 文件 | 放在 | 大小 | 来源 | 许可 | 用途 |
|---|---|---|---|---|---|
| `sam2.1_hiera_large.pt` | `week2_mask_tracking/checkpoints/` | 约 900 MB | Meta SAM 2.1：https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt | Apache-2.0 | 逐帧追踪，生成 mask |
| `yolox_s.onnx` | `week3_video_pipeline/models/` | 36 MB | OpenCV Model Zoo：https://github.com/opencv/opencv_zoo （`models/object_detection_yolox/object_detection_yolox_2022nov.onnx`，下载后改名为 `yolox_s.onnx`） | Apache-2.0（见 `models/LICENSE_yolox`） | 物体检测（COCO 80 类） |
| `big-lama.pt` | `week3_video_pipeline/models/` | 206 MB | LaMa 官方权重的 TorchScript 版：https://github.com/enesmsahin/simple-lama-inpainting （release v0.1.0） | Apache-2.0 | 补洞（背景板里从没露出的区域，以及逐帧模式） |

可选：`sam2.1_hiera_tiny.pt`，地址同上，把文件名里的 `large` 换成 `tiny`，只用于第二周的对比实验。SDXL Inpainting 的权重放在 ComfyUI 里，Week 4/5 的三个网页都不需要。

## 环境

Windows 下的一键脚本（`*.bat`）都使用 `week2_mask_tracking\.venv` 里的 Python。

```bat
cd week2_mask_tracking
py -3 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r ..\requirements.txt
set SAM2_BUILD_CUDA=0
.venv\Scripts\python -m pip install --no-build-isolation sam2==1.1.0
```

- 需要 NVIDIA GPU，开发用的是 RTX 5090 Laptop（24 GB）。CUDA 版本按自己的显卡和驱动选择，修改 PyTorch 安装命令里的 `cu130`。
- 结果视频用 Windows 自带的 Media Foundation 编码为 H.264。其他系统会退回 mp4v 编码，浏览器可能无法播放。

## 运行

| 程序 | 启动 | 地址 | 说明 |
|---|---|---|---|
| 多目标追踪 | `week5_multi_track\start_tracker.bat` | http://127.0.0.1:8768 | 上传视频，或从 `test_clips/` 选一段 |
| 视频编辑器 | `week4_video_editor\start_editor.bat` | http://127.0.0.1:8767 | 同上 |
| 演示界面 | `week4_temporal\start_demo.bat` | http://127.0.0.1:8765 | 只展示本机已经生成的结果。需要先把 `clip_A/C/D` 放进 `test_clips/`，再依次运行 `week3_video_pipeline\run_week3_video.bat` 和 `week4_temporal\run_week4.bat` |

每个任务的中间文件（抽帧、mask、结果视频）保存在各自程序目录下的 `jobs/<id>/`，这个文件夹可以随时整个删掉。
