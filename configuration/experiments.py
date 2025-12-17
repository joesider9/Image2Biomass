Layer_name_list = ['conv',
                   'conv_3d',
                   'time_distr_conv',
                   'time_distr_conv_3d',
                   'lstm',
                   'hidden_dense',
                   'dense',
                   'Flatten',
                   'Dropout']
experiments = dict()

experiments['exp_fuzzy1'] = {'row_stats': [('dense', {1})],
                             'output': [
                                 ('concatenate', {1}),
                                 ('dense', {1})
                             ],
                             }


experiments['mlp1'] = {
    'row_stats': [('dense', {2048, 1024}),
                ('dense', {'linear', 512}),
                ('dense', {256, 128, 64}),
                ],
    'row_obs': [('dense', {512, 256}),
                  ('dense', {256, 128, 64}),
                  ],
    'output': [('cross_attention', {1}),
               ('dense', {64})
               ],
}

experiments['mlp2'] = {
    'row_obs_stats': [('dense', 4096),
                ('dense', 'linear'),
                ('dense', 512),
                ],
    'output': [
        ('dense', {64})
    ],
}
#
# experiments['timm_net3'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', {128, 64}),
#                ],
#     'row_stats': [('dense', 32),
#                      ],
#     'row_obs': [('dense', 256),
#                  ('dense', 128)
#                  ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net4'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_stats': [('dense', 4),
#                  ('dense', {512, 256, 128})
#                  ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
experiments['timm_net5'] = {
    'images': [('vit_net', 1),
               ('Flatten', []),
               ('dense', 1024),
               ('dense', 256),
               ],
    'row_obs': [('dense', {512}),
                         ],
    'output': [('cross_attention', {1}),
               ('dense', 64),
               ]
}
# experiments['timm_net6'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 512),
#                ],
#     'row_obs_stats': [('dense', {2048}),
#                 ('dense', {512})
#                 ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['CrossViVit_net'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 128)
#                ],
#
#     'nwp': [('conv', {0, 1, 2, 3, 4, 5}),
#             ('conv', {0, 1, 3}),
#             ('Flatten', []),
#             ('dense', {32, 64, 128}),
#             ],
#     'row_calendar': [
#         ('dense', {32, 64, 128}),
#     ],
#     'row_obs': [
#         ('dense', {32, 64, 128}),
#     ],
#     'output': [('concatenate', 1),
#                ('dense', 64),
#                ]
# }
# experiments['Time_CrossViVit_net'] = {
#     'images': [('time_distr_vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 128)
#                ],
#     'row_calendar': [('dense', 4),
#                      ],
#     'output': [
#         ('concatenate', {1}),
#         ('dense', 64),
#     ]
# }
# experiments['trans_net'] = {
#     'lstm': [('transformer', 128),
#              ('Flatten', []),
#              ('dense', 720),
#              ('dense', 128)
#              ],
#     'output': [
#         ('concatenate', {1}),
#         ('dense', 64),
#     ]
# }
#
# experiments['yolo'] = {
#     'images': [('yolo', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     # 'row_obs_nwp': [('dense', 4),
#     #              ('dense', 720),
#     #              ('dense', 128)
#     #              ],
#     'output': [('concatenate', {1}), ('dense', 64),
#                ]
# }
# experiments['unet'] = {
#     'images': [('unet', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 64)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     # 'row_obs_nwp': [('dense', 4),
#     #              ('dense', 720),
#     #              ('dense', 128)
#     #              ],
#     'output': [('concatenate', {1}), ('dense', 256),
#                ]
# }
#
# experiments['distributed_lstm1'] = {
#     'lstm': [('lstm', 1),
#              ('Flatten', []),
#              ('dense', 0.25),
#              ('dense', 0.5),
#              ],
#     'output': [('concatenate', {1}), ('dense', 0.25),
#                ('dense', 0.5),
#                ('dense', 32)
#                ],
# }
