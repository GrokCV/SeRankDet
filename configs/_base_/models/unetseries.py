model = dict(
    name='UNet',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None
    ),
    decode_head=dict(
        type='U_Net',
        deep_supervision=True
    ),
    loss=dict(type='SoftIoULoss')
)
