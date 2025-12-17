from torch.utils.data import DataLoader
# from image_dataset_compress import ImageDatasetRealTime
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from einops import repeat, rearrange
from tqdm import tqdm
#
# def combine_dict_arrays(dict_list):
#     result = {}
#     all_keys = set().union(*dict_list)
#     for key in all_keys:
#         arrays = [d[key] for d in dict_list if key in d]
#         result[key] = torch.cat(arrays, axis=0)
#     return result
#
# def collate_fn_eval(batch):
#     fn = list(filter (lambda x:x[0] is not None if isinstance(x, tuple) else x is not None, batch))
#     dict_batch = [item[0] for item in fn]
#     timestamp_batch = [item[1] for item in fn]
#     collated_dict = combine_dict_arrays(dict_batch)
#     timestamp_tensor = torch.tensor([ts.timestamp() for ts in timestamp_batch])
#
#     return collated_dict, timestamp_tensor
#
# def collate_fn_train(batch):
#     fn = list(filter (lambda x:x[0] is not None if isinstance(x, tuple) else x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(fn)
#
#
# def feed_image_dataset_real_time(static_data, data, target, dates, params , device, batch_size=1, shuffle=False,
#                                  train=True, use_target=True, api='eumdac', is_online=False):
#     dataset = ImageDatasetRealTime(static_data, data, target, dates, params, device, train=train, use_target=use_target,
#                                    is_online=is_online, api=api)
#     dataset.create_dataset(dates)
#     dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn_train
#                                                                                         if train else collate_fn_eval)
#     return dataloader

class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 2, stride=3, padding=2), nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, 2, stride=2, padding=3), nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
#
# def train_batch(input, model, criterion, optimizer):
#     model.train()
#     optimizer.zero_grad()
#     output = model(input)
#     loss = criterion(output, input)
#     loss.backward()
#     optimizer.step()
#     return loss
#
#
# @torch.no_grad()
# def validate_batch(input, model, criterion):
#     model.eval()
#     output = model(input)
#     loss = criterion(output, input)
#     return loss

def classify_image_batch(images, static_data, sat_image_type, area_adjust, device):
    best_weights = torch.load(os.path.join(static_data['path_group'],
                                           f'autoencoder_{sat_image_type}_{area_adjust}.pt'),
                              weights_only=False,
                              map_location=torch.device('cpu'))
    torch.cuda.empty_cache()
    net_model = ConvAutoEncoder(in_channels=3 if sat_image_type == 'rgb_snow' else 1)
    net_model.load_state_dict(best_weights)
    net_model.to(device)
    net_model.eval()

    encoder_out = net_model.encoder(images.to(device)).detach().cpu().numpy()
    encoder_25 = np.quantile(encoder_out, 0.25, axis=1)
    encoder_50 = np.quantile(encoder_out, 0.5, axis=1)
    encoder_75 = np.quantile(encoder_out, 0.75, axis=1)
    encoder_25 = rearrange(encoder_25, 'b h w -> b (h w)').astype(np.float32)
    encoder_50 = rearrange(encoder_50, 'b h w -> b (h w)').astype(np.float32)
    encoder_75 = rearrange(encoder_75, 'b h w -> b (h w)').astype(np.float32)
    encoder_sum = np.concatenate((encoder_25, encoder_50, encoder_75), axis=-1)

    data_classif = encoder_sum

    file = os.path.join(static_data['path_group'], f'clusterer_{sat_image_type}_{area_adjust}.pickle')
    clusterer = joblib.load(file)['clusterer']
    return clusterer.predict(data_classif)
#
# if __name__ == "__main__":
#     from Ptolemaida.short_term.configuration.config import config
#     from eforecast.init.initialize import initializer
#     import sys
#     api = 'eumetview'
#     static_data = initializer(config())
#     static_data['sat_folder'] = 'D:/Dropbox/data_transfer' if sys.platform != 'linux' \
#         else '/media/smartrue/HHD2/Satellites/EUMETSAT/Greece'
#
#     dates = pd.date_range('2023-06-02', '2025-04-07', freq='15min')
#
#     for sat_image_type in 'ir039:ir108:vis006'.split(':'):
#         for area_adjust in [250, 125, 52]:
#             data = None
#             target = None
#             device = torch.device('cuda:0')
#             params = dict()
#             params['final_image_size'] = 224
#             params['area_adjust'] = area_adjust
#
#             params['site_coord'] = [225, 249]
#             params['image_coord'] = [43, 34, 19, 28]
#             params['image_size'] = [800, 800]
#             params['sat_image_type'] = sat_image_type
#             trn_dates, val_dates = train_test_split(dates, test_size=0.3)
#             trn_dl = feed_image_dataset_real_time(static_data, data, target, trn_dates, params , device,
#                                                    batch_size=256, shuffle=True,
#                                                    train=False, use_target=False,
#                                                    is_online=True,
#                                                    api=api)
#             val_dl = feed_image_dataset_real_time(static_data, data, target, val_dates, params , device,
#                                                    batch_size=256, shuffle=False,
#                                                    train=False, use_target=False,
#                                                    is_online=True,
#                                                    api=api)
#             model = ConvAutoEncoder(in_channels=3 if sat_image_type == 'rgb_snow' else 1).to(device)
#             from torchsummary import summary
#
#             summary(model, (3 if sat_image_type == 'rgb_snow' else 1, 224, 224))
#
#             criterion = nn.MSELoss()
#             optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
#             num_epochs = 72
#             best_val = np.inf
#             for epoch in range(num_epochs):
#                 N = len(trn_dl)
#                 for data in tqdm(trn_dl):
#                     if data[0]['images'].shape[0] == 0:
#                         continue
#                     loss = train_batch(data[0]['images'].to(device), model, criterion, optimizer)
#                 N1 = len(val_dl)
#                 l_val = 0
#                 for data in tqdm(val_dl):
#                     if data[0]['images'].shape[0] == 0:
#                         continue
#                     loss = validate_batch(data[0]['images'].to(device), model, criterion)
#                     l_val += loss
#                 l_val = l_val.detach().cpu().numpy()
#                 l_val /= len(val_dl)
#                 print(l_val)
#                 if  l_val < best_val:
#                     torch.save(model.state_dict(), os.path.join(static_data['path_group'],
#                                                                 f'autoencoder_{sat_image_type}_{area_adjust}.pt'))
#                     best_val = l_val
#             del trn_dl
#             del val_dl
#     for sat_image_type in 'ir039:ir108:vis006'.split(':'):
#         for area_adjust in [250, 125, 52]:
#             data = None
#             target = None
#             device = torch.device('cuda:0')
#             params = dict()
#             params['final_image_size'] = 224
#             params['area_adjust'] = area_adjust
#
#             params['site_coord'] = [225, 249]
#             params['image_coord'] = [43, 34, 19, 28]
#             params['image_size'] = [800, 800]
#             params['sat_image_type'] = sat_image_type
#             best_weights = torch.load(os.path.join(static_data['path_group'],
#                                                                 f'autoencoder_{sat_image_type}_{area_adjust}.pt'),
#                                       weights_only=False,
#                                       map_location=torch.device('cpu'))
#             dataset = feed_image_dataset_real_time(static_data, None, None, dates, params, 'cpu',
#                                                    batch_size=256, shuffle=True,
#                                                    train=False, use_target=False,
#                                                    is_online=True,
#                                                    api=api)
#             torch.cuda.empty_cache()
#             net_model = ConvAutoEncoder(in_channels=3 if sat_image_type == 'rgb_snow' else 1)
#             net_model.load_state_dict(best_weights)
#             net_model.to(device)
#             net_model.eval()
#             data_classif = []
#             dates_classif = pd.DatetimeIndex([])
#             for image_seq in tqdm(dataset):
#                 dates_temp = [pd.to_datetime(image_seq[1][idx].numpy(), unit='s').round('15min') for idx in
#                               range(len(image_seq[1]))]
#                 dates_classif = dates_classif.append(pd.DatetimeIndex(dates_temp))
#                 data = image_seq[0]['images']
#                 encoder_out = net_model.encoder(data.to(device)).detach().cpu().numpy()
#                 encoder_25 = np.quantile(encoder_out, 0.25, axis=1)
#                 encoder_50 = np.quantile(encoder_out, 0.5, axis=1)
#                 encoder_75 = np.quantile(encoder_out, 0.75, axis=1)
#                 encoder_25 = rearrange(encoder_25, 'b h w -> b (h w)').astype(np.float32)
#                 encoder_50 = rearrange(encoder_50, 'b h w -> b (h w)').astype(np.float32)
#                 encoder_75 = rearrange(encoder_75, 'b h w -> b (h w)').astype(np.float32)
#                 encoder_sum = np.concatenate((encoder_25, encoder_50, encoder_75), axis=-1)
#                 data_classif.append(encoder_sum)
#             data_classif = np.concatenate(data_classif, axis=0)
#             classifier = KMeans(n_clusters=96, max_iter=1000, random_state=48)
#             classifier.fit(data_classif)
#             clusters = classifier.cluster_centers_
#             idx_centers = []
#             for c in clusters:
#                 idxs = np.argsort(np.sum(np.abs(c - data_classif), axis=1))
#                 id = 0
#                 while idxs[id] in idx_centers:
#                     id += 1
#                 idx_centers.append(idxs[id])
#
#             dataset_center = feed_image_dataset_real_time(static_data, None, None,
#                                                           dates_classif[idx_centers], params,
#                                                           device,
#                                                           batch_size=96, shuffle=True,
#                                                           train=False, use_target=False,
#                                                           is_online=True,
#                                                           api=api)
#             image_seq_clust = next(iter(dataset_center))
#             data_center = image_seq_clust[0]['images'].detach().cpu().numpy()
#             data_center = (255 * data_center).astype(np.uint8)
#             dates_center = [pd.to_datetime(image_seq_clust[1][idx].numpy(), unit='s').round('15min') for idx in
#                             range(len(image_seq_clust[1]))]
#             dates_center = pd.DatetimeIndex(dates_center)
#             encode_quantiles = [np.expand_dims(np.quantile(img, [0.05, 0.25, 0.5, 0.75, 0.95]), axis=-1)
#                                 for img in data_center]
#
#             classifier_dict = {'clusterer': classifier,
#                                'clusters': data_center,
#                                'cluster_dates': dates_center,
#                                'encode_quantiles': np.concatenate(encode_quantiles, axis=-1)}
#             joblib.dump(classifier_dict, os.path.join(static_data['path_group'],
#                                                       f'clusterer_{sat_image_type}_{area_adjust}.pickle'))
#             torch.cuda.empty_cache()
