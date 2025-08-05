import os
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

import numpy as np
from PIL import Image

# def generate_inverted_soft_mask(y, thr=0.01):
#     """
#     生成一个反转的平滑过渡 mask，基于亮度信息和阈值.
    
#     :param y: 输入的多通道 PIL 图像 (通常是 RGB)
#     :param thr: 阈值，用于平滑过渡的范围
#     :return: 返回一个反转的 soft mask，格式为 PIL 图像
#     """
#     # 将 PIL 图像转换为 NumPy 数组，并归一化到 [0, 1]
#     y = np.array(y) / 255.0  # 转换为 [0, 1] 范围的浮点数数组

#     # 提取 y 的最大值（假设 y 是 RGB 图像，shape: [h, w, 3]）
#     msk = np.max(y, axis=2)  # 在通道维度上取最大值，得到单通道
#     print('msk is', msk)
    
#     # 将 msk 按照公式平滑转换为 soft mask，并进行反转
#     msk = np.where(msk>0.99, 1.0, 0.0)#np.minimum(1.0, np.maximum(0.0, (msk - 1.0 + thr) / thr))
    
#     # 调整维度以适应原始图像的形状
#     msk = np.expand_dims(msk, axis=2)  # 增加维度，使之成为 [h, w, 1]
    
#     # 将 mask 扩展到3个通道
#     msk = np.tile(msk, [1, 1, 3])
#     # msk = 1 - msk

#     # 将 mask 转换为 [0, 255] 的 uint8 格式以转换为 PIL 图像
#     msk = (msk * 255).astype(np.uint8)
#     msk_pil = Image.fromarray(msk)
    
#     return msk_pil

def generate_inverted_soft_mask(y, thr=0.01, kernel_size=3, operation='opening'):
    """
    生成一个反转的平滑过渡 mask，并进行膨胀和腐蚀操作以填充空隙或去除噪声.
    
    :param y: 输入的多通道 PIL 图像 (通常是 RGB)
    :param thr: 阈值，用于平滑过渡的范围
    :param kernel_size: 卷积核大小，默认为3
    :param operation: 操作类型 ('closing' 或 'opening')
    :return: 返回经过指定形态学操作的反转 soft mask，格式为 PIL 图像
    """
    # 将 PIL 图像转换为 NumPy 数组，并归一化到 [0, 1]
    y = np.array(y) / 255.0  # 转换为 [0, 1] 范围的浮点数数组

    # 提取 y 的最大值（假设 y 是 RGB 图像，shape: [h, w, 3]）
    msk = np.max(y, axis=2)  # 在通道维度上取最大值，得到单通道
    
    # 将 msk 按照公式平滑转换为 soft mask，并进行反转
    msk = np.where(msk > 253/255, 1.0, 0.0)  # 转换为二值化 mask

    # 定义结构化元素（核）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 选择操作类型
    if operation == 'closing':
        # 先膨胀后腐蚀 (闭操作) - 填充空隙
        msk = cv2.dilate(msk, kernel, iterations=1)
        msk = cv2.erode(msk, kernel, iterations=1)
    elif operation == 'opening':
        # 先腐蚀后膨胀 (开操作) - 去除噪声
        msk = cv2.erode(msk, kernel, iterations=1)
        msk = cv2.dilate(msk, kernel, iterations=1)

    # 调整维度以适应原始图像的形状
    msk = np.expand_dims(msk, axis=2)  # 增加维度，使之成为 [h, w, 1]
    
    # 将 mask 扩展到 3 个通道
    msk = np.tile(msk, [1, 1, 3])

    # 将 mask 转换为 [0, 255] 的 uint8 格式以转换为 PIL 图像
    msk = (msk * 255).astype(np.uint8)
    msk_pil = Image.fromarray(msk)
    
    return msk_pil


pipeline = AutoPipelineForInpainting.from_pretrained(
    "/HPS/RawDiff/work/LED/sd_model_inpainting/", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("/HPS/RawDiff/work/LED/test_results/itm_batch_rafal_all_new/42/output_image_hdr_itm_39_0.png")
# init_image = init_image/255.0
# print('init_image is', init_image)
# mask_image = 255*generate_inverted_soft_mask(init_image)#load_image("/HPS/RawDiff/work/LED/test_results/itm_batch_rafal_all_new/42/inpaint_mask.png")
mask_image = generate_inverted_soft_mask(init_image) #load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = torch.Generator("cuda").manual_seed(42)
prompt = "A tree stands alone in a field with a rocky hillside. The sky is cloudy and the sun is setting, casting a warm glow over the landscape."
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]

save_path = f'/HPS/RawDiff/work/LED/test_results/stable_diffusion_inpainting_dilate/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_name_input = os.path.join(save_path, f'orig_o.png')
save_name_mask = os.path.join(save_path, f'mask_o.png')
save_name_output = os.path.join(save_path, f'inpainting_o_new_42_opening.hdr')

init_image.save(save_name_input)
mask_image.save(save_name_mask)
# cv2.imwrite(save_name_output, cv2.cvtColor((image / 2 + 0.5).clip(0, 1)*255., cv2.COLOR_RGB2BGR))
cv2.imwrite(save_name_output, cv2.cvtColor(np.exp(image), cv2.COLOR_RGB2BGR))