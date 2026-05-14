import os
import uuid

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from flask import Flask, request, render_template_string

from model.PLGMNet import PLGMNet   # 根据你的工程结构调整导入

# -----------------------
# 一些基础配置
# -----------------------
# 输入图片缩放尺寸：建议改成你 test 代码里用的 testsize
IMG_SIZE = 384

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPLOAD_FOLDER = "static/uploads"
MASK_FOLDER = "static/masks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# -----------------------
# 与 test.py 对应的工具函数
# -----------------------
def normPRED(x):
    """
    和你原来 test.py 里的 normPRED 思路一致：
    把预测归一化到 [0, 1]
    """
    max_val = torch.max(x)
    min_val = torch.min(x)
    return (x - min_val) / (max_val - min_val + 1e-8)


transform = transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )


def load_model(weight_path="PLGMNet.pth"):
    model = PLGMNet()
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# 全局加载一次模型
MODEL = load_model("PLGMNet.pth")


def infer_single_image(img_path):
    """
    对单张图片做推理：
    输入: 图片路径
    输出: (原图 PIL, mask PIL, mask 保存路径)
    """

    # 1. 读原图
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # 2. 预处理
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

    # 3. 前向推理
    with torch.no_grad():
        outputs = MODEL(img_tensor)

        # 适配不同的返回形式（可能是 tensor，也可能是 list/tuple）
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs

        # 期望 pred 形状大致为 [B, 1, H, W] 或 [B, H, W]
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]  # 取第一个通道
        pred = pred.squeeze(0)       # 去掉 batch 维度，得到 [H, W]

        pred = normPRED(pred)
        pred = pred.to(torch.float32).cpu()

    # 4. tensor -> PIL 灰度图
    mask = transforms.ToPILImage()(pred).convert("L")
    # 如果希望和原图同大小，可以 resize 回去
    mask = mask.resize((w, h))

    # 5. 保存结果
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = base_name + "_mask.png"
    mask_path = os.path.join(MASK_FOLDER, mask_filename)
    mask.save(mask_path)

    return img, mask, mask_path


# -----------------------
# Flask Web 部分
# -----------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MASK_FOLDER"] = MASK_FOLDER

# 简单的 HTML 模板，直接写在 python 里，省得单独建 templates 文件夹
HTML_PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>PLGMNet：轻量化带钢表面缺陷检测</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        .container { max-width: 900px; margin: auto; }
        .images { display: flex; gap: 20px; margin-top: 20px; }
        .images div { text-align: center; }
        img { max-width: 400px; border: 1px solid #ddd; }
        .msg { margin-top: 10px; color: #333; }
    </style>
</head>
<body>
<div class="container">
    <h1>PLGMNet：轻量化带钢表面缺陷检测</h1>

    <form method="post" enctype="multipart/form-data">
        <p>选择一张图片：</p>
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">开始分割</button>
    </form>

    {% if error %}
    <p class="msg" style="color: red;">{{ error }}</p>
    {% endif %}

    {% if orig_url and mask_url %}
    <div class="images">
        <div>
            <h3>原图</h3>
            <img src="{{ orig_url }}">
        </div>
        <div>
            <h3>分割结果</h3>
            <img src="{{ mask_url }}">
        </div>
    </div>
    <p class="msg">分割结果已保存：{{ mask_path }}</p>
    {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    orig_url = None
    mask_url = None
    mask_path = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            error = "请先选择一张图片。"
        else:
            try:
                # 用随机前缀防止重名
                ext = os.path.splitext(file.filename)[1]
                filename = f"{uuid.uuid4().hex}{ext}"
                upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(upload_path)

                # 调用模型做推理
                _, _, mask_path = infer_single_image(upload_path)

                # 用于在页面中显示
                orig_url = "/" + upload_path.replace("\\", "/")
                mask_url = "/" + mask_path.replace("\\", "/")

            except Exception as e:
                error = f"推理出错：{e}"

    return render_template_string(
        HTML_PAGE,
        orig_url=orig_url,
        mask_url=mask_url,
        mask_path=mask_path,
        error=error
    )


if __name__ == "__main__":
    # 让外网能访问的话可以改 host="0.0.0.0"
    app.run(host="0.0.0.0", port=5000, debug=True)
