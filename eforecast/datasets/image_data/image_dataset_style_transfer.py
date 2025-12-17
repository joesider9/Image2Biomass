import os
import traceback
import cv2
import numpy as np
import pandas as pd
import joblib
import concurrent
import torch
from einops import rearrange
from einops import repeat
from skimage.metrics import structural_similarity
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, vifp
from eforecast.datasets.files_manager import FilesManager
from eforecast.datasets.image_data.image_classification import classify_image_batch

class ImageDatasetRealTime(torch.utils.data.Dataset):
    def __init__(self, static_data, data, target, dates, params, device, train=True, use_target=True, is_online=False,
                 api='eumdac', slow=True):
        self.slow = slow
        self.api = api
        self.y = None
        self.x = None
        self.use_target = use_target
        self.device = device
        self.train = train
        self.path_sat = static_data['sat_folder']
        self.horizon = static_data['horizon']
        self.static_data = static_data
        self.lat, self.long = static_data['site_indices']
        self.final_size = int(params['final_image_size'])
        self.area_adjust = params['area_adjust']
        self.image_type = params['sat_image_type'].split(':')
        self.type = static_data['type']
        self.null_classes = params['null_classes']
        # self.lags = [var_data for var_data in static_data['variables'] if var_data['type'] == 'image'][0]['lags']
        self.lags = [0]
        self.ts_resolution = static_data['ts_resolution']
        if not is_online:
            files_manager = FilesManager(static_data, train=train, is_online=is_online)
            dates_image = files_manager.check_if_exists_image_data()
            dates = dates.intersection(dates_image)
        self.dates = dates
        indices = dates.get_indexer(self.dates)

        self.init_data(data, target, indices)
        # self.spatial_coords = self.get_spatial_coords(static_data)

    def init_data(self, data, target, indices):
        if data is not None:
            self.x = dict()
            if isinstance(data, dict):
                for name in data.keys():
                    if name == 'images':
                        continue
                    if isinstance(data[name], dict):
                        self.x[name] = dict()
                        for name1 in data[name].keys():
                            values = data[name][name1][indices] if isinstance(data[name][name1], np.ndarray) \
                                else data[name][name1].values[indices]
                            self.x[name][name1] = torch.from_numpy(values)
                    else:
                        values = data[name][indices] if isinstance(data[name], np.ndarray) else data[name].values[indices]
                        self.x[name] = torch.from_numpy(values)
            else:
                self.x['input'] = torch.from_numpy(data[indices])
        if self.train:
            self.y = torch.from_numpy(target[indices]) if target is not None else None
        else:
            self.y = None

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
        return self.dates.shape[0]

    def __getitem__(self, idx):
        try:
            if self.slow:
                return self.get(idx)
            else:
                return self.get_new(idx)
        except:
            return None, None

    def get_image_grey(self, images):
        inp_lag = []
        for j in range(images.shape[0]):
            sat = images[j, :, :, :]
            sat = np.expand_dims(cv2.cvtColor(sat.astype(np.float32), cv2.COLOR_BGR2GRAY), axis=-1)
            inp_lag.append(np.expand_dims(sat, axis=0))
        return np.expand_dims(np.concatenate(inp_lag, axis=0), axis=0)

    def crop_image(self, image):
        return image[np.maximum(self.lat - self.area_adjust, 0):np.minimum(self.lat + self.area_adjust, image.shape[0]),
                      np.maximum(self.long - self.area_adjust, 0):np.minimum(self.long + self.area_adjust, image.shape[1]), :]

    def final_resize(self, images):
        image_res0 = []
        for k in range(images.shape[0]):
            image_res2 = []
            for g in range(images.shape[1]):
                img_crop = np.concatenate(
                    [np.expand_dims(cv2.resize(images[k, g, :, :, i],
                                               dsize=[self.final_size, self.final_size],
                                               interpolation=cv2.INTER_AREA), axis=-1)
                     for i in range(images.shape[-1])], -1)
                image_res2.append(img_crop)
            image_res1 = np.array(image_res2)
            image_res0.append(image_res1)
        image = np.array(image_res0)
        return image

    def data_func(self, date_):
        try:
            image_temp = []
            for image_type in self.image_type:
                path_sat = os.path.join(self.path_sat, 'jpeg', f'{date_.year}_{date_.strftime("%B")}_{date_.day}',
                                        f'{date_.hour}')
                file_sat = os.path.join(path_sat, f"HRSEVERI_{image_type}_{date_.strftime('%Y%m%dT%H%M')}.jpg")
                if not os.path.exists(file_sat):
                    continue
                a = cv2.imread(file_sat)
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if 'snow' in image_type \
                    else np.expand_dims(cv2.equalizeHist(a[..., -1]), axis=-1)
                a = self.crop_image(a)
                image_temp.append(a)
            if len(image_temp) == 0:
                return None
            return (np.expand_dims(np.concatenate(image_temp, axis=-1), axis=0), date_)
        except:
            return None

    def create_dataset(self, dates):
        image_data = []
        self.dates = []
        with concurrent.futures.ThreadPoolExecutor(19) as executor:
            result = list(executor.map(self.data_func, dates))
        for item in result:
            if item is not None:
                image_data.append(item[0])
                self.dates.append(item[1])
        if len(image_data) == 0:
            raise
        image_data = np.expand_dims(np.concatenate(image_data, axis=0), axis=0)
        image_data = self.final_resize(image_data)
        self.image_data = torch.from_numpy(rearrange(image_data, 'b t w h c -> b t c w h').astype(np.float32) / 255)[0]
        labels = classify_image_batch(self.image_data[:, 0:3, :, :], self.static_data, 'rgb_snow',  self.area_adjust,
                                      self.device)
        self.labels = labels
        self.dates = pd.DatetimeIndex(self.dates)
        if self.train:
            indices = np.where(~np.isin(labels, self.null_classes))[0]
            self.image_data = self.image_data[indices]
            self.dates = self.dates[indices]



    def get_image_eumetview(self, dates):
        try:
            image_data = []
            for date_ in dates:
                image_temp = []
                for image_type in self.image_type:
                    path_sat = os.path.join(self.path_sat, 'jpeg', f'{date_.year}_{date_.strftime("%B")}_{date_.day}',
                                            f'{date_.hour}')
                    file_sat = os.path.join(path_sat, f"HRSEVERI_{image_type}_{date_.strftime('%Y%m%dT%H%M')}.jpg")
                    if not os.path.exists(file_sat):
                        continue
                    a = cv2.imread(file_sat)
                    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if 'snow' in image_type \
                            else np.expand_dims(cv2.equalizeHist(a[..., -1]), axis=-1)
                    a = self.crop_image(a)
                    image_temp.append(a)
                if len(image_temp) == 0:
                    continue
                image_data.append(np.expand_dims(np.concatenate(image_temp, axis=-1), axis=0))
        except:
            raise
        try:
            if len(image_data) == 0:
                raise
            image_data = np.expand_dims(np.concatenate(image_data, axis=0), axis=0)
            image_data = self.final_resize(image_data)
            x_img =torch.from_numpy(rearrange(image_data, 'b t w h c -> b t c w h').astype(np.float32) / 255)
        except:
            raise
        return x_img

    def get_image_eumdac(self, dates):
        try:
            image_data = []
            for date_ in dates:
                file_sat = os.path.join(self.path_sat, 'processed',
                                        f'satellite_{date_.strftime("%Y_%m_%d__%H_%M")}.pkl')
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

    def get_data(self, idx):
        x_data = dict()
        if isinstance(self.x, dict):
            for name in self.x.keys():
                if isinstance(self.x[name], dict):
                    x_data[name] = dict()
                    for name1 in self.x[name].keys():
                        x_data[name][name1] = self.x[name][name1][idx].float().to(self.device)
                else:
                    x_data[name] = self.x[name][idx].float().to(self.device)
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
        except:
            raise
        dates_obs = pd.DatetimeIndex([date + pd.DateOffset(hours=l) for l in self.lags][::-1])
        dates_pred = pd.date_range(date, date + pd.DateOffset(hours=self.horizon), freq=self.ts_resolution)
        try:
            if self.api == 'eumdac':
                x_img_obs = self.get_image_eumdac(dates_obs)
            else:
                x_img_obs = self.get_image_eumetview(dates_obs)
        except:
            raise
        images = x_img_obs[0].float().to(self.device)
        if images.shape[1] != 6:
            raise
        data = images[:, 3:6, :, :].float().to(self.device)
        target = images[:, 0:3, :, :].float().to(self.device)
        return data, target

    def get_new(self, idx):
        data = self.image_data[idx][None, 3:-1, :, :].float().to(self.device)
        target = self.image_data[idx][None, :3, :, :].float().to(self.device)
        labels = self.labels[idx]
        date = self.dates[idx]
        if self.train:
            return data, target
        else:
            return data, target, labels, date