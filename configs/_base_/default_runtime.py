# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = False
random_seed = 42
gpus = 1
