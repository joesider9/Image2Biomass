import os
import cv2
import joblib
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
import numpy as np
import pandas as pd

class DatasetImageStatsCreator:

    def __init__(self, static_data, transformer, dates=None, train=False, parallel=False, refit=False):
        self.refit = refit
        self.static_data = static_data
        self.transformer = transformer
        self.train = train
        self.parallel = parallel
        ts_res = str.lower(static_data['ts_resolution'])
        self.dates = dates

        self.path_sat = static_data['path_image']
        self.path_sat_processed = static_data['path_image']
        self.n_jobs = static_data['n_jobs']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'image'])
        self.apis = set([var_data['source'] for var_data in self.variables.values()])
        for var in self.variables.keys():
            if var in self.transformer.variables_index.keys():
                self.transformer.fit(np.array([]), var, data_dates=dates)
        self.coord = static_data['coord']
        print(f"Dataset Image Stats creation started for project {self.static_data['_id']}")


    def make_dataset(self):
        if not self.parallel:
            stats = []
            for t in tqdm(self.dates):
                if os.path.exists(os.path.join(self.path_sat_processed, 'train' if self.train else 'test',
                                                   f'{t}.jpg')):
                    res = self.process(t)
                    if res is None:
                        continue
                    data = res
                    stats.append(data)
                else:
                    continue
        else:
            data = Parallel(n_jobs=18)(
                delayed(self.process)(t) for t in tqdm(self.dates)
                                                            if os.path.exists(os.path.join(self.path_sat_processed,
                                                                                           'train' if self.train else 'test',
                                                                                            f'{t}.jpg')))
            stats = [d for d in data if d is not None]
        if len(stats) > 0:
                    stats = pd.concat(stats)
        return stats

    def process(self, date):
        path_image = os.path.join(self.path_sat, 'train' if self.train else 'test')
        file = os.path.join(path_image, f'{date}.jpg')
        x = cv2.imread(file)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x_3d = self.create_stats(date, x)
        return x_3d

    def create_green_yellow_mask_hsv(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([120, 255, 255])

        lower_yellow = np.array([30, 50, 50])
        upper_yellow = np.array([40, 255, 255])

        lower_white = np.array([220, 50, 50])
        upper_white = np.array([255, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(green_mask, yellow_mask, white_mask)

        return combined_mask.astype(bool)

    def create_stats(self, tag, img):
        img = img.astype(np.uint8)
        mask = self.create_green_yellow_mask_hsv(img)
        img_masked = img.copy()
        img_masked[~mask] = 0
        colour_ratio = [np.sum(mask) / np.prod(img.shape[:2])]
        columns = ['colour_ratio']
        colour_quantiles = []
        for i ,c in enumerate('HSV'):
            columns.extend(f'img_masked_{c}_{p}' for p in [5, 25, 50, 75, 95])
            colour_quantiles.extend(list(np.percentile(img_masked[:, :, i], [5, 25, 50, 75, 95])))
        for i ,c in enumerate('HSV'):
            columns.extend(f'img_{c}_{p}' for p in [5, 25, 50, 75, 95])
            colour_quantiles.extend(list(np.percentile(img[:, :, i], [5, 25, 50, 75, 95])))
        return pd.DataFrame([colour_ratio + colour_quantiles], index=[tag], columns=columns)


