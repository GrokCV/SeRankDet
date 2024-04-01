_base_ = [
    '../_base_/datasets/sirstaug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_500e.py',
    '../_base_/models/unetseries.py'
]

model = dict(
    decode_head=dict(
        type='SeRankDet'
    )
)

optimizer = dict(
    type='AdamW',
    setting=dict(lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
)
runner = dict(type='EpochBasedRunner', max_epochs=300)
data = dict(
    train_batch=8,
    test_batch=8)
develop = dict(source_file_root='/data1/ppw/works/All_ISTD/model/UNetSeries/SeRankDet.py')
