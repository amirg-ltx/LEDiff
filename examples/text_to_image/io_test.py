import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
import numpy as np
# import natsort
import re

class CustomDataset(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        jsonl_file = os.path.join(self.path, 'metadata.jsonl')
        with open(jsonl_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                self.data.append((data['file_name'], data['text']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        # img_path = natsort.natsorted(img_path)
        image_name = os.path.join(self.path, img_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', image_name)
        image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED) # cv2.imread(image_name)#
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的图像是 BGR 格式，需要转换为 RGB
        
        mu = 100
        if image.dtype == np.uint8:
            # 8位图像
            max_val = 255
            image = (image/max_val)
            # image = np.log(1 + mu*image)/np.log(1 + mu)
            image = image*2.0 - 1.0
            # print('max val is', max_val)
            
        elif image.dtype == np.uint16:
            # 16位图像
            max_val = 65535
            image = image/max_val
            # image = np.log(1 + mu*image)/np.log(1 + mu)
            image = image*2.0 - 1.0
        else:
            # print("name is", image_name)
            # print('max is', np.max(image))
            # print('min is', np.min(image))
            # image = np.log(1 + mu*image)/np.log(1 + mu)
            image = image*2.0 - 1.0
            
            #raise ValueError("Unsupported image depth: {}".format(image.dtype))
            
        
        # print("max val is", max_val)
        # image = (image/max_val)*2.0 - 1.0
        image = cv2.resize(image, (512, 512))
        # image = 2*(image/255) - 1.0#2*(image/65525) - 1.0
        # image = np.array(image, dtype = np.float16)
        # image = cv2.resize(image, (512, 512))

        if self.transform:
            image = self.transform(image)

        if self.text_transform:
            # print('caption is', caption)
            caption = self.text_transform(caption)

        return image, caption
    
  
class CustomDatasetImage(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.jpg', '.jpeg', '.tif', '.tiff', '.png', '.hdr', '.exr']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        image_name = os.path.join(self.path, img_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', image_name)
        image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED) # cv2.imread(image_name)#
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取的图像是 BGR 格式，需要转换为 RGB
        # print('image.dtype is', image.dtype)
        if image.dtype == np.uint8:
            # 8位图像
            max_val = 255
            image = (image/max_val)
            # image = np.log(1 + mu*image)/np.log(1 + mu)
            image = image*2.0 - 1.0
            # print('max val is', max_val)
            print('lllllll')
            
        elif image.dtype == np.uint16:
            # 16位图像
            max_val = 65535
            image = image/max_val
            # image = np.log(1 + mu*image)/np.log(1 + mu)
            image = image*2.0 - 1.0
        else:
            # print("name is", image_name)
            # print('max is', np.max(image))
            # print('min is', np.min(image))
            # image = np.log(1 + mu*image)/np.log(1 + mu)
            image = image*2.0 - 1.0
            
        # print('io_test max is, min is', np.max(image), np.min(image))
        # print('io_test mean is', np.mean(image))
   
        height, width = image.shape[:2]

        # # 计算新的尺寸，向下取整到最近的16的倍数
        # new_width = (width // 16) * 16
        # new_height = (height // 16) * 16

        # # 调整图像大小
        # image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # if height > 256 or width > 256:
            # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            
        # height, width = image.shape[:2]
            
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        # 调整图像大小
        if height > 512 or width > 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        # image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if self.transform:
            image = self.transform(image)
        
        # print('io_test tensor max is, min tensor is', torch.max(image), torch.min(image))
        # print('io_test mean is', torch.mean(image))

        return image
    
    
class CustomDatasetNPZ(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.npz']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        file_name = os.path.join(self.path, file_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', file_name)
        data = np.load(file_name)
        
        ldr_1, ldr_2, ldr_3, hdr = data['jpg1'], data['jpg2'], data['jpg3'], data['hdr']
        
        ldr_1 = (ldr_1/255.0)*2.0 - 1.0
        ldr_2 = (ldr_2/255.0)*2.0 - 1.0
        ldr_3 = (ldr_3/255.0)*2.0 - 1.0
        # if np.min(hdr) < 0.0:
        #     hdr = hdr - np.min
        hdr = np.where(hdr < 0.0, 0.0, hdr)
        # hdr = 1.0 * hdr/np.mean(hdr)
        # hdr = np.clip(hdr, 0.0, 60000.0)         
        # print('io_test max is, min is', np.max(image), np.min(image))
        # print('io_test mean is', np.mean(image))

        if self.transform:
            ldr_1 = self.transform(ldr_1)
            ldr_2 = self.transform(ldr_2)
            ldr_3 = self.transform(ldr_3)    
            hdr = self.transform(hdr)    
        # print('io_test tensor max is, min tensor is', torch.max(image), torch.min(image))
        # print('io_test mean is', torch.mean(image))

        return [ldr_1, ldr_2, ldr_3, hdr]

class CustomDatasetNPZ_five(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.npz']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        file_name = os.path.join(self.path, file_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', file_name)
        data = np.load(file_name)
        
        ldr_1, ldr_2, ldr_3, ldr_4, ldr_5, hdr = data['jpg1'], data['jpg2'], data['jpg3'], data['jpg4'], data['jpg5'], data['hdr']
        
        ldr_1 = (ldr_1/255.0)*2.0 - 1.0
        ldr_2 = (ldr_2/255.0)*2.0 - 1.0
        ldr_3 = (ldr_3/255.0)*2.0 - 1.0
        ldr_4 = (ldr_4/255.0)*2.0 - 1.0
        ldr_5 = (ldr_5/255.0)*2.0 - 1.0

        # if np.min(hdr) < 0.0:
        #     hdr = hdr - np.min
        hdr = np.where(hdr < 0.0, 0.0, hdr)
        # hdr = 1.0 * hdr/np.mean(hdr)
        # hdr = np.clip(hdr, 0.0, 60000.0)         
        # print('io_test max is, min is', np.max(image), np.min(image))
        # print('io_test mean is', np.mean(image))

        if self.transform:
            ldr_1 = self.transform(ldr_1)
            ldr_2 = self.transform(ldr_2)
            ldr_3 = self.transform(ldr_3) 
            ldr_4 = self.transform(ldr_4)
            ldr_5 = self.transform(ldr_5)  
            hdr = self.transform(hdr)    
        # print('io_test tensor max is, min tensor is', torch.max(image), torch.min(image))
        # print('io_test mean is', torch.mean(image))

        return [ldr_1, ldr_2, ldr_3, ldr_4, ldr_5, hdr]

    
class CustomDatasetNPZCaption(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.npz']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        file_name = os.path.join(self.path, file_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        print('image_name is', file_name)
        data = np.load(file_name)
        
        ldr_1, ldr_2, ldr_3, caption1, caption2,  caption3 = data['jpg1'], data['jpg2'], data['jpg3'], data['caption1'].item(), data['caption2'].item(), data['caption3'].item()
        print('caption 1 is', data['caption1'].item())
        exit()

        ldr_1 = (ldr_1/255.0)*2.0 - 1.0
        ldr_2 = (ldr_2/255.0)*2.0 - 1.0
        ldr_3 = (ldr_3/255.0)*2.0 - 1.0
        # hdr = 1.0 * hdr/np.mean(hdr)
        # hdr = np.clip(hdr, 0.0, 60000.0)         
        # print('io_test max is, min is', np.max(image), np.min(image))
        # print('io_test mean is', np.mean(image))

        if self.transform:
            ldr_1 = self.transform(ldr_1)
            ldr_2 = self.transform(ldr_2)
            ldr_3 = self.transform(ldr_3)    
        if self.text_transform:
            caption1 = self.text_transform(caption1)
            caption2 = self.text_transform(caption2)
            caption3 = self.text_transform(caption3)

        return [ldr_1, ldr_2, ldr_3, caption1, caption2,  caption3, caption1, caption2,  caption3]
    
def split_and_extract(text):
    # 使用正则表达式匹配可能为负数的曝光值
    match = re.search(r"The exposure value is: (-?\d+\.?\d*)", text)
    if match:
        exposure_value = float(match.group(1))  # 提取并转换为浮点数
    else:
        exposure_value = None  # 如果没有找到，设置为None

    # 使用第一个句号进行分割
    parts = text.split('.', 1)  # 只分割一次
    if len(parts) > 1:
        first_part = parts[0] + '.'  # 包含句号
        second_part = parts[1].strip()  # 删除句号后的空格
    else:
        first_part = text
        second_part = ''

    return first_part, second_part, exposure_value
    
    
def remove_numbered_prefix(caption):
    # 使用正则表达式去掉前面的数字和句号
    return re.sub(r'^\d+\.\s*', '', caption)
        
class CustomDatasetNPZCaption_Exp(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.npz']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        file_name = os.path.join(self.path, file_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', file_name)
        data = np.load(file_name)
        
        ldr_1, ldr_2, ldr_3, caption_1_all, caption_2_all,  caption_3_all = data['jpg1'], data['jpg2'], data['jpg3'], data['caption1'].item(), data['caption2'].item(), data['caption3'].item()
        # print('caption 1 is', data['caption1'].item())
        
        ldr_1 = (ldr_1/255.0)*2.0 - 1.0
        ldr_2 = (ldr_2/255.0)*2.0 - 1.0
        ldr_3 = (ldr_3/255.0)*2.0 - 1.0
        # print('caption_1_all is', caption_1_all)
        # print('caption_2_all is', caption_2_all)
        # print('caption_3_all is', caption_3_all)
        
        _, caption1, exposure_1 = split_and_extract(caption_1_all)
        _, caption2, exposure_2 = split_and_extract(caption_2_all)
        _, caption3, exposure_3 = split_and_extract(caption_3_all)
        caption1 = remove_numbered_prefix(caption1)
        caption2 = remove_numbered_prefix(caption2)
        caption3 = remove_numbered_prefix(caption3)
        
        # print('caption1 is', caption1)
        # print('exposure_1 is', exposure_1)
        # print('caption2 is', caption2)
        # print('exposure_2 is', exposure_2)
        # print('caption3 is', caption3)
        # print('exposure_3 is', exposure_3)
        # exit()

        if self.transform:
            ldr_1 = self.transform(ldr_1)
            ldr_2 = self.transform(ldr_2)
            ldr_3 = self.transform(ldr_3)    
            exposure_1 = torch.tensor(exposure_1)
            exposure_2 = torch.tensor(exposure_2)
            exposure_3 = torch.tensor(exposure_3)
            
        if self.text_transform:
            caption1 = self.text_transform(caption1)
            caption2 = self.text_transform(caption2)
            caption3 = self.text_transform(caption3)

        return [ldr_1, ldr_2, ldr_3, caption1, caption2, caption3, exposure_1, exposure_2, exposure_3]


class CustomDatasetNPZCaption_Exp_token(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.npz']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        file_name = os.path.join(self.path, file_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', file_name)
        data = np.load(file_name)
        
        ldr_1, ldr_2, ldr_3, caption_1_all, caption_2_all,  caption_3_all = data['jpg1'], data['jpg2'], data['jpg3'], data['caption1'].item(), data['caption2'].item(), data['caption3'].item()
        # print('caption 1 is', data['caption1'].item())
        # exit()
        ldr_1 = (ldr_1/255.0)*2.0 - 1.0
        ldr_2 = (ldr_2/255.0)*2.0 - 1.0
        ldr_3 = (ldr_3/255.0)*2.0 - 1.0
        # print('caption_1_all is', caption_1_all)
        # print('caption_2_all is', caption_2_all)
        # print('caption_3_all is', caption_3_all)
        
        _, caption1, _ = split_and_extract(caption_1_all)
        _, caption2, _ = split_and_extract(caption_2_all)
        _, caption3, _ = split_and_extract(caption_3_all)

        caption1 = remove_numbered_prefix(caption1)
        caption2 = remove_numbered_prefix(caption2)
        caption3 = remove_numbered_prefix(caption3)
        
        # print('caption1 is', caption1)
        # print('exposure_1 is', exposure_1)
        # print('caption2 is', caption2)
        # print('exposure_2 is', exposure_2)
        # print('caption3 is', caption3)
        # print('exposure_3 is', exposure_3)
        # exit()

        if self.transform:
            ldr_1 = self.transform(ldr_1)
            ldr_2 = self.transform(ldr_2)
            ldr_3 = self.transform(ldr_3)    
            # exposure_1 = torch.tensor(exposure_1)
            # exposure_2 = torch.tensor(exposure_2)
            # exposure_3 = torch.tensor(exposure_3)
            
        if self.text_transform:
            caption1 = self.text_transform(caption1)
            caption2 = self.text_transform(caption2)
            caption3 = self.text_transform(caption3)
            exposure_1 = self.text_transform("low")
            exposure_2 = self.text_transform("medium")
            exposure_3 = self.text_transform("high")

        return [ldr_1, ldr_2, ldr_3, caption1, caption2, caption3, exposure_1, exposure_2, exposure_3]

class CustomDatasetNPZCaption_Exp_token_five(Dataset):
    def __init__(self, path, transform=None, text_transform=None):
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.text_transform = text_transform
        self.data = []
        self.path = path
        extensions=['.npz']
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 将文件的完整路径添加到列表中
                # all_files.append(os.path.join(root, file))
                if any(file.lower().endswith(ext) for ext in extensions):
                    # 将文件的完整路径添加到列表中
                    self.data.append((os.path.join(root, file)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        # img_path = natsort.natsorted(img_path)

        file_name = os.path.join(self.path, file_path)#os.path.join(self.path, img_path.replace('jpg', 'tif'))
        # print('image_name is', file_name)
        data = np.load(file_name)
        
        ldr_1, ldr_2, ldr_3, ldr_4, ldr_5, caption = data['jpg1'], data['jpg2'], data['jpg3'], data['jpg4'], data['jpg3'], data['caption1'].item()
        # print('caption 1 is', data['caption1'].item())
        # exit()
        ldr_1 = (ldr_1/255.0)*2.0 - 1.0
        ldr_2 = (ldr_2/255.0)*2.0 - 1.0
        ldr_3 = (ldr_3/255.0)*2.0 - 1.0
        ldr_4 = (ldr_4/255.0)*2.0 - 1.0
        ldr_5 = (ldr_5/255.0)*2.0 - 1.0
        # print('caption_1_all is', caption_1_all)
        # print('caption_2_all is', caption_2_all)
        # print('caption_3_all is', caption_3_all)
        
        _, caption1, _ = split_and_extract(caption)
        # _, caption2, _ = split_and_extract(caption_2_all)
        # _, caption3, _ = split_and_extract(caption_3_all)

        caption1 = remove_numbered_prefix(caption1)
        # caption2 = remove_numbered_prefix(caption2)
        # caption3 = remove_numbered_prefix(caption3)
        
        # print('caption1 is', caption1)
        # print('exposure_1 is', exposure_1)
        # print('caption2 is', caption2)
        # print('exposure_2 is', exposure_2)
        # print('caption3 is', caption3)
        # print('exposure_3 is', exposure_3)
        # exit()

        if self.transform:
            ldr_1 = self.transform(ldr_1)
            ldr_2 = self.transform(ldr_2)
            ldr_3 = self.transform(ldr_3)
            ldr_4 = self.transform(ldr_4)
            ldr_5 = self.transform(ldr_5)     
            # exposure_1 = torch.tensor(exposure_1)
            # exposure_2 = torch.tensor(exposure_2)
            # exposure_3 = torch.tensor(exposure_3)
            
        if self.text_transform:
            caption1 = self.text_transform(caption1)
            # caption2 = self.text_transform(caption2)
            # caption3 = self.text_transform(caption3)
            # exposure_1 = self.text_transform("low")
            # exposure_2 = self.text_transform("medium")
            # exposure_3 = self.text_transform("high")

        return [ldr_1, ldr_2, ldr_3, ldr_4, ldr_5, caption1]

