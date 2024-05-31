from mmengine.config import read_base

# The new config inherits a base config to highlight the necessary modification
with read_base():
    from .._base_.default_runtime import *
    from .._base_.models.cascade_rcnn_r50_fpn import *

# We also need to change the num_classes in head to match the dataset's annotation
model.merge(
    dict(
        roi_head=dict(
            bbox_head=[
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.05, 0.05, 0.1, 0.1],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
                ),
                dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1,
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.033, 0.033, 0.067, 0.067],
                    ),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                    ),
                    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
                ),
            ]
        )
    )
)


# Modify dataset related settings
data_root = "../../data/dataset/"
backend_args = None
metainfo = dict(
    classes=("tamgi"),
    palette=[(220, 20, 60)],
)

train_pipeline = [  # Training data processing pipeline
    dict(
        type="LoadImageFromFile", backend_args=backend_args
    ),  # First pipeline to load images from file path
    dict(
        type="LoadAnnotations",  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=False,  # Whether to use instance mask, True for instance segmentation
        poly2mask=False,
    ),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
    dict(
        type="Resize",  # Pipeline that resizes the images and their annotations
        scale=(960,),  # The largest scale of the images
        keep_ratio=True,  # Whether to keep the ratio between height and width
    ),
    dict(
        type="RandomFlip",  # Augmentation pipeline that flips the images and their annotations
        prob=0.5,
    ),  # The probability to flip
    dict(
        type="PackDetInputs"
    ),  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]

test_pipeline = [  # Testing data processing pipeline
    dict(
        type="LoadImageFromFile", backend_args=backend_args
    ),  # First pipeline to load images from file path
    dict(
        type="Resize", scale=(1333, 800), keep_ratio=True
    ),  # Pipeline that resizes the images
    dict(
        type="PackDetInputs",  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="detection_labels/coco_annotations_train.json",
        data_prefix=dict(img=""),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="detection_labels/coco_annotations_val.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="detection_labels/coco_annotations_test.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
# Modify metric related settings
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "detection_labels/coco_annotations_val.json",
    metric=["bbox"],
    format_only=False,
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "detection_labels/coco_annotations_test.json",
    metric=["bbox"],
    format_only=False,
)

optim_wrapper = dict(
    # Define the optimizer configuration
    optimizer=dict(
        type="Adam",  # Type of optimizer
        lr=1e-3,  # Learning rate
        weight_decay=0.0001,
    ),
    # Specify model parameters to optimize
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(
                lr_mult=0.1, decay_mult=1.0
            ),  # Specific multiplier for backbone parameters
            "head": dict(lr_mult=1.0),  # Multiplier for head parameters
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]
train_cfg = dict(
    by_epoch=True,  # set by_epoch=True or type='EpochBasedTrainLoop'
    max_epochs=50,
    val_interval=2,
)
val_cfg = dict(type="ValLoop", dataloader=val_dataloader, evaluator=val_evaluator)
test_cfg = dict(type="TestLoop", dataloader=test_dataloader, evaluator=test_evaluator)
