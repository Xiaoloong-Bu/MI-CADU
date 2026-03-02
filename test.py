from PIL import Image
import numpy as np
import torch
from CADU import CAUD
import os
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import argparse


to_tensor = transforms.Compose([transforms.ToTensor()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()


# 不同类型的参数
parser.add_argument('--control', default=0, type=int, help='控制执行方式：0-人工引导需要指定hu_guid，1-自适应融合，2-不再引导')
parser.add_argument('--hu_guid', default=2, type=int, help='人工引导-选定类别，对应cls_dct中的键值')
parser.add_argument('--save_path', default=r"", help='融合图像保存路径')
parser.add_argument('--data_path', default=r"", help='数据集加载路径-测试文件夹包括：ir和vi两个子文件夹，文件夹文件要求对应图像同名')
parser.add_argument('--model_path', default=r"./CADU-97796-model.pt", help='预训练模型路径')
args = parser.parse_args()

# 类别
cls_dct = {'ir_Low_contrast': 0, 'ir_Random_noise': 1, 'ir_Stripe_noise': 2, 'vis_Blur': 3, 'vis_Exposure': 4, 'vis_far_near': 5,
'vis_Haze': 6, 'vis_Low-Light': 7, 'vis_mutil_exposure': 8, 'vis_Rain': 9, 'vis_Random_noise': 10}

save_path = args.save_path
data_path = args.data_path
model_path = args.model_path
control = args.control
target = args.hu_guid

if not os.path.exists(save_path):
    os.makedirs(save_path)

model = CAUD(6, 32)
model.to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
summary = sum([np.prod(param.shape) for param in model.parameters()])
print(f"模型参数量：{summary}")
model.eval()

with torch.no_grad():
    # 记录分类正确的数量
    cls_pre_num = 0
    # 记录各种分类类别的数量
    cls_pre_lst = [0 for _ in range(11)]
    for img_name in tqdm(os.listdir(data_path + r"\ir")):
        img_name = img_name
        inf_path = os.path.join(data_path, "ir", img_name)
        vis_path = os.path.join(data_path, "vi", img_name)
        inf_img = Image.open(inf_path).convert("RGB")
        vis_img = Image.open(vis_path).convert("RGB")

        height, width = inf_img.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        inf_img = inf_img.resize((new_height, new_width))
        vis_img = vis_img.resize((new_height, new_width))

        inf_img = to_tensor(inf_img)
        vis_img = to_tensor(vis_img)

        # 输入模型的数据处理
        inf_img = inf_img.unsqueeze(0)
        vis_img = vis_img.unsqueeze(0)
        inf_img = inf_img.to(device)
        vis_img = vis_img.to(device)

        # 图像融合
        # ------------------------------------------------------------------------------------------------------------
        labels = torch.LongTensor([1])
        num_classes = 11
        noise_std = 0.1  # 噪声标准差
        cls_f_lst = [target, ]
        one_hot_list = [
            (F.one_hot(labels * cls, num_classes=num_classes).float() * 50 + torch.randn_like(
                F.one_hot(labels * cls, num_classes=num_classes).float()) * noise_std).to(device)
            for i, cls in enumerate(cls_f_lst)
        ]
        if control == 0:
            out, cls = model(inf_img, vis_img, one_hot_list)
            sign = ""  # 用于我判断哪些图像是分类正确，哪些是分类错误的融合结果
            cls_pre_lst[cls.argmax().item()] += 1
            if cls.argmax().item() == target:
                cls_pre_num += 1
                sign += '+'
            else:
                sign += '-'
        elif control == 1:
            out, cls = model(inf_img, vis_img)
        elif control == 2:
            out, cls = model(inf_img, vis_img, use_cls=False)
        # ------------------------------------------------------------------------------------------------------------
        out = out[0].permute(1, 2, 0).detach()
        out = out.cpu().clamp(0, 1).numpy()
        out = (out - out.min()) / (out.max() - out.min()) * 255
        rgb_fused_img = Image.fromarray(out.round().astype(np.uint8))

        # 融合结果合成彩色图像
        if control == 0:
            output_path = os.path.join(save_path, img_name.split(".")[0] + sign + "." + img_name.split(".")[1])
        else:
            output_path = os.path.join(save_path, img_name.split(".")[0] + "." + img_name.split(".")[1])
        # 保存
        rgb_fused_img.save(output_path)
print(f"分类准确率：{cls_pre_num / len(os.listdir(os.path.join(data_path, 'ir'))):.4f}")
print(f"具体分类结果，真值：{target}, 分类结果：{cls_pre_lst}")