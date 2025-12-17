import os
import traceback
import cv2
import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from einops import repeat

class ImageDatasetRealTime(torch.utils.data.Dataset):
    def __init__(self, static_data, data, target, dates, params, train=True, use_target=True):
        self.y = None
        self.x = None
        self.use_target = use_target
        self.train = train
        self.static_data = static_data
        self.final_size = 224
        self.type = static_data['type']
        self.path_image = static_data['path_image']
        self.dates = dates
        self.init_data(data, target)
        self.params = params
        self.static_data = static_data
        self.transforming = self.init_transforms()
        self.init_mask()

    def init_mask(self):
        self.masks = dict()
        path_image = os.path.join(self.path_image, 'train' if self.train else 'test')
        for date in self.dates:
            if not os.path.exists(os.path.join(path_image, f'mask_{date}.pkl')):
                file = os.path.join(path_image, f'{date}.jpg')
                x = cv2.imread(file)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                mask = self.create_green_yellow_mask_hsv(x)
                joblib.dump(mask, os.path.join(path_image, f'mask_{date}.pkl'))

    def init_transforms(self):
        # if self.use_target:
        #     return transforms.Compose([
        #         transforms.RandomCrop(224),
        #     ])
        # else:

        return self.divide_image_to_patches
    
    def divide_image_to_patches(self, image):
        C, H, W = image.shape
        patches = []

        patch_size = 224

        n_patches_h = 4
        n_patches_w = 8
        start_point_h = 1000 - 4 * 224
        start_point_w = 2000 - 8 * 224
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                start_h = start_point_h + i * patch_size
                start_w = start_point_w + j * patch_size
                end_h = start_h + patch_size
                end_w = start_w + patch_size

                patch = image[:, start_h:end_h, start_w:end_w]
                
                patches.append(patch)
        
        return patches

    def init_data(self, data, target):
        if data is not None:
            self.x = dict()
            if isinstance(data, dict):
                for name in data.keys():
                    if name == 'images':
                        continue
                    if isinstance(data[name], dict):
                        self.x[name] = dict()
                        for name1 in data[name].keys():
                            values = data[name][name1] if isinstance(data[name][name1], np.ndarray) \
                                else data[name][name1].values
                            self.x[name][name1] = torch.from_numpy(values)
                    else:
                        values = data[name] if isinstance(data[name], np.ndarray) else data[name].values
                        self.x[name] = torch.from_numpy(values)
            else:
                self.x['input'] = torch.from_numpy(data)
        if self.train:
            self.y = torch.from_numpy(target) if target is not None else None
        else:
            self.y = None


    def __len__(self) -> int:
        return self.dates.shape[0]

    def __getitem__(self, idx):
        try:
            return self.get(idx)
        except:
            return None, None

    def create_green_yellow_mask_hsv(self, img):
        # Convert RGB to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define HSV ranges for green and yellow
        # Green in HSV: Hue ~60-120, Saturation >50, Value >50
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([120, 255, 255])

        # Yellow in HSV: Hue ~20-40, Saturation >50, Value >50
        lower_yellow = np.array([30, 50, 50])
        upper_yellow = np.array([40, 255, 255])

        # Yellow in HSV: Hue ~20-40, Saturation >50, Value >50
        lower_white = np.array([220, 50, 50])
        upper_white = np.array([255, 255, 255])

        # Create masks for each color
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Combine masks
        combined_mask = cv2.bitwise_or(green_mask, yellow_mask, white_mask)

        return combined_mask.astype(float)[np.newaxis, ...]

    def get_image_grey(self, images):
        inp_lag = []
        for j in range(images.shape[0]):
            sat = images[j, :, :, :]
            sat = np.expand_dims(cv2.cvtColor(sat.astype(np.float32), cv2.COLOR_BGR2GRAY), axis=-1)
            inp_lag.append(np.expand_dims(sat, axis=0))
        return np.expand_dims(np.concatenate(inp_lag, axis=0), axis=0)

    def get_image(self, date, use_patches=False, patch_idx=None):
        try:
            path_image = os.path.join(self.path_image, 'train' if self.train else 'test')
            file = os.path.join(path_image, f'{date}.jpg')
            x = cv2.imread(file)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(x)
            mask = joblib.load(os.path.join(path_image, f'mask_{date}.pkl'))
            mask = torch.from_numpy(mask)
            x = rearrange(x, 'h w c -> c h w')
            if x is None:
                return None
        except:
            print('Cannot find image with tag: ', date)
            return None
            
        # Combine image and mask
        combined = torch.cat([x / 255, mask], axis=0)

        x = self.transforming(combined)

        return x


    def get_data(self, idx):
        x_data = dict()
        if isinstance(self.x, dict):
            for name in self.x.keys():
                if isinstance(self.x[name], dict):
                    x_data[name] = dict()
                    for name1 in self.x[name].keys():
                        x_data[name][name1] = self.x[name][name1][idx].float()
                else:
                    x_data[name] = self.x[name][idx].float()
        else:
            raise ValueError('Input must be dict')
        return x_data

    def get(self, idx):
        date = self.dates[idx]
        try:
            if self.x is not None:
                X = self.get_data(idx)
            else:
                X = None
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            raise
        try:
            x_img_obs = self.get_image(date)
            # if isinstance(x_img_obs, list):
            x_img_obs = torch.stack(x_img_obs)
            n = x_img_obs.shape[0]
            if self.use_target:
                torch.randint(0, n, (4,))
                x_img_obs = x_img_obs[torch.randint(0, n, (4,))]
                X['row_stats'] = repeat(X['row_stats'].unsqueeze(0), '1 ... -> n ...', n=4)
            else:
                X['row_stats'] = repeat(X['row_stats'].unsqueeze(0), '1 ... -> n ...', n=n)
                date = [date for _ in range(n)]

        except Exception as e:
            tb = traceback.format_exception(e)
            if 'FileNotFoundError' not in "".join(tb):
                print("".join(tb))
            raise
        return_tensors = {
            "images": x_img_obs.float()
        }
        if X is not None:
            return_tensors.update(X)
        if self.use_target:
            y = torch.cat([self.y[idx].unsqueeze(0).float() for _ in range(4)])
            return return_tensors, y
        else:
            return return_tensors, date

