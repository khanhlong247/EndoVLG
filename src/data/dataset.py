import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
import random
from pathlib import Path
from torchvision import transforms

class EndoDataset(Dataset):
    def __init__(self, data_dir, json_path, tokenizer=None, mode='train', max_regions=6):
        self.data_dir = Path(data_dir)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.mode = mode 
        
        self.global_size = (256, 256)
        self.texture_size = (256, 256)
        self.geometry_size = (256, 256)
        self.max_regions = max_regions 
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def generate_edge(self, mask):
        """Generate edge ground truth from the segmentation mask"""
        mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
        edge = cv2.Laplacian(mask_np, cv2.CV_64F)
        edge = np.absolute(edge)
        edge = (edge > 0).astype(np.float32) 
        return edge

    def augment_pair(self, img, mask, aggressive=False):
        if self.mode != 'train': return img, mask
        h, w = img.shape[:2]
        if random.random() < 0.5:
            img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            img = cv2.flip(img, 0); mask = cv2.flip(mask, 0)
        angle = random.uniform(-30, 30)
        scale = random.uniform(0.7, 1.3) 
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101) 
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if random.random() < 0.8:
            alpha = random.uniform(0.8, 1.2); beta = random.uniform(-30, 30)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img, mask

    def resize_pad(self, img, target_size, is_mask=False):
        h, w = img.shape[:2]
        scale = min(target_size[0]/h, target_size[1]/w)
        nh, nw = int(h*scale), int(w*scale)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (nw, nh), interpolation=interp)
        
        if len(img.shape) == 3:
            new_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            new_img = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
            
        y_off = (target_size[0] - nh) // 2
        x_off = (target_size[1] - nw) // 2
        if len(img.shape) == 3:
            new_img[y_off:y_off+nh, x_off:x_off+nw, :] = img_resized
        else:
            new_img[y_off:y_off+nh, x_off:x_off+nw] = img_resized
        return new_img

    def get_auto_boxes(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        generated_regions = []
        img_h, img_w = mask.shape
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 10
            generated_regions.append({"box": [max(0,x-pad), max(0,y-pad), min(img_w,x+w+pad), min(img_h,y+h+pad)], "label_text": "lesion"})
        return generated_regions

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item.get('image', f"images/{item['id']}.jpg")
        img_path = self.data_dir / img_name
        
        image = cv2.imread(str(img_path))
        if image is None: return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Logic to find the mask based on the image ID
        file_base = item['id']
        mask_path = os.path.join(self.data_dir, "masks", file_base + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.data_dir, "masks", file_base + ".jpg")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.data_dir, "masks", file_base + ".jpeg")
            
        mask_color = cv2.imread(mask_path)
        if mask_color is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.max(mask_color, axis=2)
            mask = (mask > 10).astype(np.uint8) * 255

        # 1. STAGE 1: GLOBAL STREAM (kept as-is for Stage 1/2)
        aug_global_img, aug_global_mask = self.augment_pair(image.copy(), mask.copy())
        
        global_img_np = cv2.resize(aug_global_img, self.global_size, interpolation=cv2.INTER_NEAREST)
        global_tensor = self.normalize(global_img_np)
        
        global_mask_np = cv2.resize(aug_global_mask, self.global_size, interpolation=cv2.INTER_NEAREST)
        global_mask_tensor = self.mask_transform(global_mask_np) 

        global_edge_np = self.generate_edge(global_mask_tensor)
        global_edge_tensor = torch.from_numpy(global_edge_np).unsqueeze(0)

        # 2. STAGE 2 & 3: LOCAL STREAMS + TEXT
        texture_list, geometry_list = [], []
        regions = item.get('regions', [])
        if len(regions) == 0 and mask.sum() > 0:
            regions = self.get_auto_boxes((mask > 127).astype(np.uint8))
            
        if self.mode == 'train': random.shuffle(regions) 
        
        for region in regions:
            x1, y1, x2, y2 = region['box']
            crop_img = image[int(y1):int(y2), int(x1):int(x2)]
            crop_mask = mask[int(y1):int(y2), int(x1):int(x2)]
            if crop_img.size == 0: continue

            aug_crop_img, aug_crop_mask = self.augment_pair(crop_img, crop_mask)
            
            # Texture Stream
            tex_final_np = cv2.resize(aug_crop_img, self.texture_size, interpolation=cv2.INTER_LINEAR)
            texture_list.append(self.normalize(tex_final_np))
            
            # Geometry Stream
            geo_final_np = cv2.resize(aug_crop_mask, self.geometry_size, interpolation=cv2.INTER_NEAREST)
            geometry_list.append(self.mask_transform(geo_final_np).repeat(3, 1, 1))
            
        # Pad the empty regions
        num_valid_regions = min(len(texture_list), self.max_regions)
        texture_tensor = torch.zeros((self.max_regions, 3, *self.texture_size))
        geometry_tensor = torch.zeros((self.max_regions, 3, *self.geometry_size))
        
        for i in range(num_valid_regions):
            texture_tensor[i] = texture_list[i]
            geometry_tensor[i] = geometry_list[i]
            
        question_text = item.get('question', "")
        answer_text = item.get('answer', item.get('text', ""))
            
        return {
            "global": global_tensor,         
            "global_mask": global_mask_tensor, 
            "global_edge": global_edge_tensor,
            "texture": texture_tensor,       
            "geometry": geometry_tensor,    
            "question": question_text,
            "answer": answer_text,
            "num_regions": num_valid_regions,
            "regions": regions
        }