_base_ = [
    "../_base_/models/SwinUNETR.py",
    "../_base_/datasets/slovenia_224x224.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
class_weight = None

model = dict(
    backbone=dict(
        type="SwinUNETR",img_size=(16, 224, 224), in_channels=2, out_channels=8
    ),
    decode_head=dict(
        type="PseudoHead", num_classes=8,loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
    ),
    # model training and testing settings
    train_cfg = dict(),
    test_cfg = dict(_delete_=True, mode="slide", crop_size=(224, 224), stride=(200, 200))
    )