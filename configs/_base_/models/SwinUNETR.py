# model settings
# norm_cfg = dict(type="BN", requires_grad=True)
# conv_cfg = dict(type="Conv3d")
# norm_cfg3d = dict(type="BN3d", requires_grad=True)
# patch_size = [2, 4, 4]
# img_size = [12, 224, 224]
# block_cls='ISABlock' # todo TwinsBlock
pad_size = (224, 224)
data_preprocessor = dict(
    type='MulSegDataPreProcessor',
    bgr_to_rgb=True,
    size=pad_size,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="SwinUNETR",img_size=(16, 224, 224), in_channels=8, out_channels=8
    ),
    decode_head=dict(
        type="PseudoHead", num_classes=8 ,loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
    ),
    # model training and testing settings
    train_cfg = dict(),
    test_cfg = dict(_delete_=True, mode="slide", crop_size=(224, 224), stride=(200, 200))
    )