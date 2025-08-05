import torch
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np

model_path = "/sensei-fs/users/wchao/Codes/diffusers/resutls/sd-adobe-all-own-dataaset-model/"#"/sensei-fs/users/wchao/Codes/diffusers/resutls/sd-adobe-all-model/"#"CompVis/stable-diffusion-v1-4"#
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="This image shows a boy's bedroom. Inside the bedroom, there is a window with bright sunlight streaming in from outside, indicating that it's likely afternoon. Inside the room, there is a bed, a desk, and shadows cast by the sunlight. The contrast between the light outside and the light inside is very high.").images[0] 
#pipe(prompt="The image shows a display of fresh vegetables, primarily consisting of green onions or scallions, arranged in bunches. The onions are hanging from strings, creating an appealing and organized presentation. The display is set up in a market or grocery store, showcasing the fresh produce for sale. The onions are arranged in various bunches, with some placed closer to the front and others further back, creating a visually pleasing arrangement for customers.").images[0]


# 确保图像是16位
# if image.mode != 'I;16':
#     print('llllllll')
#     image = image.convert('I;16')

# 将图像转换为 NumPy 数组
image_array = image + 0.5#np.min(image) #np.array(image)

# 将 NumPy 数组保存为 .hdr 文件
output_hdr_path = '/sensei-fs/users/wchao/Codes/diffusers/resutls/output_image_ours_window_no_clamp_own.hdr'
cv2.imwrite(output_hdr_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

print(f"Image saved as {output_hdr_path}")
# image.save("adobe_test_orig.png")
# print('Done!')

# image = pipe(prompt="yoda").images[0]
# image.save("yoda-naruto.png")

# from diffusers import StableDiffusionPipeline
# import torch

# device = "cuda"

# # load model
# model_path = "/sensei-fs/users/wchao/Codes/diffusers/examples/text_to_image/sd-adobe-model/"
# pipe = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     torch_dtype=torch.float16,ß
#     safety_checker=None,
#     feature_extractor=None,
#     requires_safety_checker=False
# )

# # load lora weights
# pipe.unet.load_attn_procs(model_path, subfolder="checkpoint-3000")
# pipe.to("cuda")

# image = pipe(prompt="The image shows a display of fresh vegetables, primarily consisting of green onions or scallions, arranged in bunches. The onions are hanging from strings, creating an appealing and organized presentation. The display is set up in a market or grocery store, showcasing the fresh produce for sale. The onions are arranged in various bunches, with some placed closer to the front and others further back, creating a visually pleasing arrangement for customers.").images[0]
# image.save("adobe_test.png")