_base_ = [
    '../_base_/datasets/nuaa.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/unetseries.py'
]

model = dict(
    decode_head=dict(
        type='SeRankDet',
        deep_supervision=True
    )
)

optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
)

runner = dict(type='EpochBasedRunner', max_epochs=1500)
data = dict(
    train_batch=4,
    test_batch=4)
develop = dict(source_file_root='/data1/ppw/works/All_ISTD/model/UNetSeries/SeRankDet.py')
find_unused_parameters = True
random_seed = 42
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=25642 rebuild_train.py configs/unetseries/unetseries_serankdet_512x512_500e_irstd1k.py
# python rebuild_train.py configs/unetseries/unetseries_serankdet_512x512_800e_nuaa.py
