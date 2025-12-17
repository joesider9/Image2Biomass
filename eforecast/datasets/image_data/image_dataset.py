import os
from tqdm import tqdm
import random
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
import torchvision.transforms.functional as F
from einops import rearrange
from einops import repeat

class ImageDataloader:
    def __init__(self, dataset, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = len(self.dataset.cv_mask)
        self.n_batches = int(self.length / self.batch_size) + 1
        self.dates = set()
        self.valid = True

    def reset(self):
        self.dates = set()
        self.valid = True

    def get_batch(self, device=torch.device('cpu'), randomly=True):
        random.seed(random.randint(1, 100))
        dates = set(self.dataset.cv_mask).difference(self.dates)
        if len(dates) == 0:
            self.valid = False
        if len(dates) > self.batch_size:
            if randomly:
                dates = random.sample(dates, k=self.batch_size)
            else:
                dates = set(list(dates)[:self.batch_size ])
        self.dates = self.dates.union(dates)

        batch = [self.dataset.get_safe(date) for date in tqdm(dates)]

        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            self.valid = False
        batch_dict = dict()
        dates_list = []
        for sample in batch:
            if sample is not None:
                for key in sample.keys():
                    if key not in batch_dict.keys() and key != 'date':
                        batch_dict[key] = []
                    if key == 'date':
                        dates_list.append(sample[key])
                    else:
                        batch_dict[key].append(sample[key].unsqueeze(0))

        for key in batch_dict.keys():
            batch_dict[key] = torch.cat(batch_dict[key], 0).to(device)
        batch_dict['dates'] = pd.DatetimeIndex(dates_list)
        return batch_dict



class ImageDataset:
    def __init__(self, static_data, cv_mask, params, use_target=True, api='eumdac'):
        self.api = api
        self.final_size = int(params['final_image_size'])
        self.use_target = use_target
        self.path_sat = static_data['sat_folder']
        self.horizon = static_data['horizon']
        self.static_data = static_data
        self.lat, self.long = static_data['site_indices']
        self.area_adjust = params['area_adjust']
        self.image_type = params['sat_image_type'].split(':')
        self.type = static_data['type']
        # self.lags = [var_data for var_data in static_data['variables'] if var_data['type'] == 'image'][0]['lags']
        self.lags = [0]
        self.ts_resolution = static_data['ts_resolution']
        self.cv_mask = cv_mask

    def get_spatial_coords(self, params):
        site_coord = params['coord']
        site_coord = np.expand_dims(np.array(site_coord), (1, 2))
        spatial_coord = params['image_coord']
        image_size = params['image_size']
        lat = np.linspace(spatial_coord[0], spatial_coord[1], image_size[0])
        lon = np.linspace(spatial_coord[2], spatial_coord[3], image_size[1])
        spatial_coords = np.stack(np.meshgrid(lon, lat)[::-1], axis=0)
        site_coord = repeat(site_coord, 'n w h -> n (w k) (h m)', k=spatial_coords.shape[1],
                            m=spatial_coords.shape[2])

        data = spatial_coords - site_coord
        data = np.power(data[0], 2) + np.power(data[1], 2)
        data = data[(None,) * 3 + (...,)]
        data = repeat(data, 'b t c w h -> b (t k) c w h', k=len(self.lags))
        spatial_coord_3d = data[:, :, :,
                           self.lat - self.area_adjust:self.lat + self.area_adjust,
                           self.long - self.area_adjust:self.long + self.area_adjust]
        spatial_coord_3d = spatial_coord_3d.squeeze()
        spatial_coord_3d = rearrange(spatial_coord_3d, 'c w h -> w h c')
        spatial_coord_3d = np.concatenate(
            [np.expand_dims(cv2.resize(spatial_coord_3d[:, :, i],
                                       dsize=[self.final_size, self.final_size],
                                       interpolation=cv2.INTER_CUBIC), axis=-1)
             for i in range(spatial_coord_3d.shape[-1])], -1)
        spatial_coord_3d = rearrange(spatial_coord_3d, 'w h c -> 1 1 w h c')
        spatial_coord_3d = self.final_resize(spatial_coord_3d)
        return spatial_coord_3d.astype(np.float32)

    def __len__(self) -> int:
        return self.cv_mask.shape[0]

    def get_safe(self, date):
        try:
            return self.get(date)
        except:
            return None

    def final_resize(self, images):
        image_res0 = []
        for k in range(images.shape[0]):
            image_res2 = []
            for g in range(images.shape[1]):
                img_crop = np.concatenate(
                    [np.expand_dims(cv2.resize(images[k, g, :, :, i],
                                               dsize=[self.final_size, self.final_size],
                                               interpolation=cv2.INTER_CUBIC), axis=-1)
                     for i in range(images.shape[-1])], -1)
                image_res2.append(img_crop)
            image_res1 = np.array(image_res2)
            image_res0.append(image_res1)
        image = np.array(image_res0)
        return image

    def upscaled_image(self, sr, image):
        return np.concatenate(
            [np.expand_dims(sr.upsample(image[:, :, i]), axis=-1) for i in range(image.shape[-1])], -1)

    def get_image_grey(self, images):
        inp_lag = []
        for j in range(images.shape[0]):
            sat = images[j, :, :, :]
            sat = np.expand_dims(cv2.cvtColor(sat.astype(np.float32), cv2.COLOR_BGR2GRAY), axis=-1)
            inp_lag.append(np.expand_dims(sat, axis=0))
        return np.expand_dims(np.concatenate(inp_lag, axis=0), axis=0)


    def size_64x64(self, images):
        image_res0 = []
        for k in range(images.shape[0]):
            image_res2 = []
            for g in range(images.shape[1]):
                img_crop = np.concatenate(
                    [np.expand_dims(cv2.resize(images[k, g, :, :, i],
                                               dsize=[64, 64],
                                               interpolation=cv2.INTER_CUBIC), axis=-1)
                     for i in range(images.shape[-1])], -1)
                image_res2.append(img_crop)
            image_res1 = np.array(image_res2)
            image_res0.append(image_res1)
        image = np.array(image_res0)
        return image

    def get_image_eumdac(self, dates):
        try:
            image_data = []
            for date_ in dates:
                file_sat = os.path.join(self.path_sat, 'processed', f'satellite_{date_.strftime("%Y_%m_%d__%H_%M")}.pkl')
                image_data.append(joblib.load(file_sat))
        except Exception as e:
            # print('Cannot find images for date {}'.format(date))
            raise e
        try:
            image_data = {k: np.concatenate([i[k] for i in image_data], axis=1) for k in image_data[0].keys()}


            x_img1 = []
            for img_tag in self.image_type:
                img_var = img_tag.split('_')[0]
                if 'coord' in img_var:
                    img = self.spatial_coords
                else:
                    img = image_data[img_var]

                    if 'grey' in img_tag:
                        img = self.get_image_grey(img[0])
                        img = self.final_resize(img)
                    else:
                        img = self.final_resize(img)

                img = torch.from_numpy(rearrange(img, 'b t w h c -> b t c w h'))
                x_img1.append(img)

            x_img = torch.cat(x_img1, 2)
        except:
            print('Something went wrong')
            raise
        return x_img

    def get_image_eumetview(self, dates):
        try:
            image_data = []
            for date_ in dates:
                image_temp = []
                for image_type in self.image_type:
                    path_sat = os.path.join(self.path_sat, 'jpeg', f'{date_.year}_{date_.strftime("%B")}_{date_.day}',
                                            f'{date_.hour}')
                    file_sat = os.path.join(path_sat, f"HRSEVERI_{image_type}_{date_.strftime('%Y%m%dT%H%M')}.jpg")
                    a = cv2.imread(file_sat)
                    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                    image_temp.append(a) if 'snow' in image_type \
                        else image_temp.append(np.expand_dims(a[...,-1], axis=-1))
                image_data.append(np.expand_dims(np.concatenate(image_temp, axis=-1), axis=0))
        except Exception as e:
            # print('Cannot find images for date {}'.format(date))
            raise e
        try:
            image_data = np.expand_dims(np.concatenate(image_data, axis=0), axis=0)
            x_img =torch.from_numpy(rearrange(image_data, 'b t w h c -> b t c w h'))
        except:
            print('Something went wrong')
            raise
        return x_img

    def get(self, date):
        dates_obs = pd.DatetimeIndex([date + pd.DateOffset(hours=l) for l in self.lags][::-1])
        dates_pred = pd.date_range(date, date + pd.DateOffset(hours=self.horizon), freq=self.ts_resolution)
        try:
            if self.api == 'eumdac':
                x_img_obs = self.get_image_eumdac(dates_obs)
            else:
                x_img_obs = self.get_image_eumetview(dates_obs)
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            raise
        return_tensors = {
            "date": pd.DatetimeIndex([date]),
            "images": x_img_obs[0].float(),
        }
        if self.use_target:
            try:
                if self.api == 'eumdac':
                    x_img_pred = self.get_image_eumdac(dates_pred)
                else:
                    x_img_pred = self.get_image_eumetview(dates_pred)
            except:
                print('Something went wrong')
                raise
            return_tensors["target"] = torch.from_numpy(x_img_pred).float()
        return return_tensors
