CHECKPOINT_DIR = "results/trained_weights/normalized_flow_weights/_checkpoints_mold"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
    "mold",
    "Cast_reference",
    "Connector_reference"
]

BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 10
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1
ANOM_MAPS_LOCATION = 'anom_maps'
