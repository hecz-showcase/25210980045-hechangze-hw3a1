# 基于 3DGS 与 AIGC 的多源资产生成与真实场景融合

本项目完成了一个基于 3D Gaussian Splatting 与 AIGC 的多源 3D 资产生成与场景融合任务。项目包含三类 3D 资产生成路径：文本到 3D 生成、单图到 3D 生成、真实场景 3DGS 重建，并将这些资产用于最终的统一场景融合与渲染展示。

本仓库主要包含以下部分：

* 物体 B：基于 threestudio + Stable Diffusion + SDS Loss 的文本到 3D 生成。
* 物体 C：基于 Stable Zero123 的单图到 3D 生成。
* 背景场景：基于 Mip-NeRF 360 `garden` 场景的 3D Gaussian Splatting 重建。
* 结果可视化：导出多视角渲染图、环绕视频、Mesh 资产和 3DGS 背景重建结果。
* 实验报告材料：训练日志、渲染结果、对比图和关键输出文件说明。

> 说明：物体 A（手机多视角真实物体 + COLMAP + 3DGS）可按本 README 中 3DGS 背景重建流程迁移完成。本文档重点总结当前已完成的文本到 3D、单图到 3D 和背景场景重建部分。

---

## 1. 项目结构

推荐的仓库结构如下：

```text
.
├── README.md
├── report/
│   ├── figures/
│   │   ├── text_to_3d/
│   │   ├── image_to_3d/
│   │   └── background_3dgs/
│   └── final_report.pdf
├── scripts/
│   ├── download_mipnerf360_garden.py
│   ├── make_video_from_renders.sh
│   └── export_mesh.sh
├── assets/
│   ├── input_images/
│   │   └── dog_rgba.png
│   ├── text_to_3d_hamburger/
│   ├── image_to_3d_dog/
│   └── background_garden/
├── logs/
│   ├── text_to_3d/
│   ├── image_to_3d/
│   └── background_3dgs/
└── docs/
    └── environment_notes.md
```

实际训练中使用了两个主要代码仓库：

```text
~/autodl-tmp/threestudio
~/autodl-tmp/gaussian-splatting
```

---

## 2. 环境配置

### 2.1 threestudio 环境

本项目使用 threestudio 完成文本到 3D 和单图到 3D 生成。由于原始依赖较老，部署时需要注意 Hugging Face 生态版本兼容问题。

推荐核心版本：

```text
Python == 3.8
torch == 2.0.0 / 2.0.0+cu118
diffusers == 0.19.3
transformers == 4.28.1
huggingface_hub == 0.19.3
accelerate == 0.20.3
xformers == 0.0.20
lightning == 2.0.0
```

安装流程示例：

```bash
cd ~/autodl-tmp

git clone https://github.com/threestudio-project/threestudio.git
cd threestudio

pip install -U pip setuptools wheel ninja
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 根据 requirements.txt 安装主要依赖
pip install --no-build-isolation -r requirements.txt
```

如果出现 `diffusers` 和 `huggingface_hub` 不兼容，执行：

```bash
pip uninstall -y huggingface_hub accelerate
pip install huggingface_hub==0.19.3 accelerate==0.20.3
```

如果访问 Hugging Face 较慢，可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 2.2 3DGS 环境

背景场景重建使用 GraphDECO 官方 `gaussian-splatting` 仓库。

```bash
cd ~/autodl-tmp

git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
```

由于官方 `environment.yml` 较老，本实验没有直接使用 `conda env create -f environment.yml`，而是在现有 CUDA/PyTorch 环境中手动安装核心扩展：

```bash
cd ~/autodl-tmp/gaussian-splatting

git submodule update --init --recursive
pip install plyfile opencv-python tqdm joblib ninja

pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e submodules/simple-knn
pip install --no-build-isolation -e submodules/fused-ssim
```

如果 `diff-gaussian-rasterization` 编译时报错：

```text
fatal error: glm/glm.hpp: No such file or directory
```

需要手动补充 `glm`：

```bash
cd ~/autodl-tmp/gaussian-splatting

rm -rf submodules/diff-gaussian-rasterization/third_party/glm

git clone --depth=1 https://github.com/g-truc/glm.git \
    submodules/diff-gaussian-rasterization/third_party/glm

pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
```

验证环境：

```bash
python -c "import torch; import gaussian_renderer; from scene import Scene; print('3DGS OK')"
```

---

## 3. 数据准备

### 3.1 文本到 3D 输入

文本到 3D 不需要图像输入，只需要 Prompt。本实验使用：

```text
a delicious hamburger
```

对应生成物体为汉堡。

---

### 3.2 单图到 3D 输入

单图到 3D 使用一张去除背景后的真实物体图片。本实验使用的输入图片为：

```text
load/images/dog_rgba.png
```

图片要求：

* 单个物体。
* 背景去除，最好为透明背景 PNG。
* 物体居中，占画面主体。
* 推荐尺寸为 512×512 左右。

如果使用在线工具去背景，处理后将图片上传到：

```text
~/autodl-tmp/threestudio/load/images/
```

---

### 3.3 Stable Zero123 权重

Stable Zero123 需要单独下载权重文件：

```bash
cd ~/autodl-tmp/threestudio/load/zero123

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download stabilityai/stable-zero123 \
    stable_zero123.ckpt \
    --local-dir .
```

下载完成后目录应类似：

```text
load/zero123/
├── download.sh
├── sd-objaverse-finetune-c_concat-256.yaml
└── stable_zero123.ckpt
```

如果 `stable_zero123.ckpt` 显示为符号链接，属于正常情况，实际文件位于 Hugging Face 缓存目录中。

---

### 3.4 Mip-NeRF 360 Garden 场景

背景场景选择 Mip-NeRF 360 中的 `garden` 场景。为了避免下载完整数据集，本实验使用 Hugging Face 数据集中的单场景目录下载方式。

创建下载脚本：

```python
# scripts/download_mipnerf360_garden.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mileleap/mipnerf360",
    repo_type="dataset",
    allow_patterns=["garden/**"],
    local_dir="./360_v2",
    local_dir_use_symlinks=False,
    resume_download=True
)
```

运行：

```bash
cd ~/autodl-tmp/mipnerf360
python scripts/download_mipnerf360_garden.py
```

最终数据目录为：

```text
~/autodl-tmp/mipnerf360/360_v2/garden/
├── images/
├── images_2/
├── images_4/
├── images_8/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── poses_bounds.npy
```

---

## 4. 训练与测试命令

### 4.1 文本到 3D：DreamFusion + SDS

首先将 `configs/dreamfusion-sd.yaml` 中默认模型替换为 Stable Diffusion 1.5：

```yaml
prompt_processor:
  pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"

guidance:
  pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"
```

训练命令：

```bash
cd ~/autodl-tmp/threestudio

export HF_ENDPOINT=https://hf-mirror.com

python launch.py \
    --config configs/dreamfusion-sd.yaml \
    --train \
    --gpu 0 \
    system.prompt_processor.prompt="a delicious hamburger"
```

本实验训练 10000 steps，输出目录示例：

```text
outputs/dreamfusion-sd/a_delicious_hamburger@20260613-205732/
├── ckpts/
│   ├── epoch=0-step=10000.ckpt
│   └── last.ckpt
├── save/
│   ├── it10000-0.png
│   ├── it10000-test/
│   └── it10000-test.mp4
├── configs/
├── csv_logs/
└── tb_logs/
```

结果查看：

```bash
ls outputs/dreamfusion-sd/a_delicious_hamburger@*/save
```

推荐放入报告的结果：

```text
save/it10000-0.png
save/it10000-test.mp4
save/it10000-test/*.png
csv_logs/version_0/metrics.csv
tb_logs/version_0/
```

导出 Mesh：

```bash
TRIAL_DIR=outputs/dreamfusion-sd/a_delicious_hamburger@20260613-205732

python launch.py \
    --config ${TRIAL_DIR}/configs/parsed.yaml \
    --export \
    --gpu 0 \
    resume=${TRIAL_DIR}/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter \
    system.exporter.context_type=cuda \
    system.geometry.isosurface_method=mc-cpu \
    system.geometry.isosurface_resolution=256 \
    system.geometry.isosurface_threshold=10.
```

导出后查找模型：

```bash
find ${TRIAL_DIR} -name "*.obj" -o -name "*.mtl" -o -name "*.png"
```

---

### 4.2 单图到 3D：Stable Zero123

训练命令：

```bash
cd ~/autodl-tmp/threestudio

python launch.py \
    --config configs/stable-zero123.yaml \
    --train \
    --gpu 0 \
    data.image_path=./load/images/dog_rgba.png
```

如果出现显存不足，可降低分辨率与采样数：

```bash
python launch.py \
    --config configs/stable-zero123.yaml \
    --train \
    --gpu 0 \
    data.image_path=./load/images/dog_rgba.png \
    data.width=128 \
    data.height=128 \
    system.renderer.num_samples_per_ray=128
```

训练完成后输出目录一般位于：

```text
outputs/zero123-sai/dog_rgba@时间戳/
├── ckpts/
├── save/
├── configs/
└── tb_logs/
```

导出 Mesh：

```bash
TRIAL_DIR=outputs/zero123-sai/dog_rgba@时间戳

python launch.py \
    --config ${TRIAL_DIR}/configs/parsed.yaml \
    --export \
    --gpu 0 \
    resume=${TRIAL_DIR}/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter \
    system.exporter.context_type=cuda \
    system.geometry.isosurface_method=mc-cpu \
    system.geometry.isosurface_resolution=256 \
    system.geometry.isosurface_threshold=10.
```

推荐放入报告的结果：

```text
load/images/dog_rgba.png
outputs/zero123-sai/dog_rgba@*/save/*.png
outputs/zero123-sai/dog_rgba@*/save/*test.mp4
导出的 dog mesh 截图
```

---

### 4.3 背景场景 3DGS 重建：Mip-NeRF 360 Garden

训练命令：

```bash
cd ~/autodl-tmp/gaussian-splatting

python train.py \
    -s ~/autodl-tmp/mipnerf360/360_v2/garden \
    -m output/garden \
    --eval
```

训练完成后主要输出：

```text
output/garden/
├── input.ply
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply
│   └── iteration_30000/
│       └── point_cloud.ply
├── train/
│   └── ours_30000/
│       ├── renders/
│       └── gt/
└── test/
    └── ours_30000/
        ├── renders/
        └── gt/
```

其中：

* `input.ply`：COLMAP 初始化稀疏点云。
* `iteration_7000/point_cloud.ply`：第 7000 步 Gaussian 模型。
* `iteration_30000/point_cloud.ply`：最终 Gaussian 模型。
* `renders/`：3DGS 模型生成的彩色渲染图。
* `gt/`：原始真实图像，用于对比。

渲染训练/测试视角：

```bash
python render.py \
    -m output/garden
```

渲染结果通常位于：

```text
output/garden/train/ours_30000/renders/
output/garden/train/ours_30000/gt/
output/garden/test/ours_30000/renders/
output/garden/test/ours_30000/gt/
```

合成测试视角视频：

```bash
ffmpeg -framerate 30 \
    -i output/garden/test/ours_30000/renders/%05d.png \
    -c:v libx264 -pix_fmt yuv420p \
    output/garden_render_video.mp4
```

注意：`render.py` 默认按照数据集原始相机位姿逐帧渲染，主要用于质量评估，不一定形成平滑的漫游轨迹。最终展示视频建议在 Blender 中设置平滑 Camera Path，或使用专门的 3DGS Viewer / 自定义轨迹渲染脚本。

---

## 5. 场景融合与漫游渲染

本项目将三个前景物体 A、B、C 插入到重建得到的 3DGS 背景场景中，并生成多视角漫游渲染视频。其中，物体 A 和物体 B 为 threestudio 生成后导出的 `.obj` Mesh 模型，物体 C 和背景场景为 3D Gaussian Splatting 形式的 `.ply` 高斯点云。

由于 Mesh 和 3DGS 属于不同的三维表示形式，本项目没有将所有资产强行转换为同一种几何格式，而是采用混合表示与合成渲染的方式完成融合。背景 3DGS 和物体 C 通过 KIRI 3DGS Render 插件导入 Blender，两个 `.obj` 物体则作为普通 Mesh 模型导入 Blender。所有资产统一放置在 Blender 的世界坐标系中，通过平移、旋转和缩放调整三个前景物体的比例、朝向和空间位置，使其与背景场景在视觉上保持一致。

在融合过程中，背景 3DGS 被固定为全局空间参考，三个前景物体根据场景中的地面、桌面和空间尺度进行人工对齐，避免出现悬空、穿模或比例不合理的问题。对于 Mesh 物体，主要调整其顶点模型的整体变换；对于 3DGS 物体，则调整其高斯点云对象及对应代理对象的整体变换。

最终渲染时保留两类资产的原始表达形式：3DGS 背景和 3DGS 物体由 KIRI 插件进行高斯渲染，OBJ Mesh 物体由 Blender 原生渲染管线渲染。为了将两类结果合成为同一画面，在 KIRI 3DGS Render 的 Render 模式中开启 `Combine With Native Render`，从而把高斯渲染层与 Blender 原生 Mesh 渲染层合成到最终帧中。由于 OBJ 物体依赖 Blender 灯光和材质显示，场景中额外添加了区域光和太阳光，并检查了 OBJ 的材质与贴图，避免 Mesh 物体在合成结果中变成黑影。

漫游视频通过 Blender 摄像机关键帧实现。首先在场景中添加 Camera，然后分别设置整体远景、物体 A、物体 B、物体 C 和最终整体展示等多个视角，并在对应帧插入摄像机的位置与旋转关键帧。Blender 会自动在关键帧之间插值，形成连续的多视角漫游路径。

渲染输出使用 KIRI 3DGS Render 插件完成。首先将 Blender 输出格式设置为 PNG 图像序列，然后在 KIRI 的 Render 模式中设置输出目录，开启 `Render Animation` 和 `Combine With Native Render`。插件会根据摄像机动画逐帧渲染合成结果，输出连续 PNG 图片序列。最后再将图片序列编码为 `.mp4`，得到最终的多视角漫游渲染视频。



---
## 6. 常见问题

### Q1: 为什么 3DGS 的 `point_cloud.ply` 导入 Blender 后只有黑色背景和很多点？

因为该文件不是传统 Mesh，而是 Gaussian primitives 的参数集合。Blender 默认把它当普通 PLY 点云导入，因此只能看到高斯中心点，无法得到 3DGS 原生渲染器中的照片级效果。正确的展示方式是使用 3DGS 的 `render.py` 或 Viewer 渲染彩色图像。

### Q2: `render.py` 生成的视频为什么不像平滑漫游？

`render.py` 默认渲染原始 train/test 相机位姿，这些相机来自数据集拍摄路径，不是人为设计的平滑相机轨迹。因此直接拼接会出现跳变。最终展示视频建议使用 Blender 相机路径或自定义相机轨迹生成。

### Q3: 为什么文本到 3D 从 SD2.1 改成 SD1.5？

原配置默认使用 `stabilityai/stable-diffusion-2-1-base`，但在实际部署中存在访问权限和下载问题。实验改用 `runwayml/stable-diffusion-v1-5`，不改变 SDS 优化流程，仍可完成文本到 3D 生成。

### Q4: 为什么 Stable Zero123 下载很慢？

`stable_zero123.ckpt` 约 8.6GB，服务器访问 Hugging Face 或镜像站可能较慢。可使用：

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

然后重新执行 `huggingface-cli download`。

---

## 7. 致谢

本项目基于以下开源项目完成：

* threestudio：用于文本到 3D 和单图到 3D 生成。
* GraphDECO 3D Gaussian Splatting：用于背景场景 3DGS 重建。
* Stable Diffusion：用于文本条件扩散先验。
* Stable Zero123 / Zero123：用于单图到 3D 生成。
* Mip-NeRF 360：用于真实背景场景数据。
