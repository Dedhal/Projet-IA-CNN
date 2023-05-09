
import fiftyone as fo
import fiftyone.zoo as foz

#
# Load 50 random samples from the validation split
#
# Only the required images will be downloaded (if necessary).
# By default, only detections are loaded
#

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="test",
    max_samples=2500,
    shuffle=True,
    dataset_dir="G:\Datasets\CoCo\\biclass"
)

session = fo.launch_app(dataset)

input("Press Enter to continue...")

#
# Load segmentations for 25 samples from the validation split that
# contain cats and dogs
#
# Images that contain all `classes` will be prioritized first, followed
# by images that contain at least one of the required `classes`. If
# there are not enough images matching `classes` in the split to meet
# `max_samples`, only the available images will be loaded.
#
# Images will only be downloaded if necessary
#

#   airplane = 5
#   apple = 52
#   horse = 19
#   tie = 32
#   toothbrush = 90
#   person = 1
#   [5, 52, 19, 1, 90]

# cat = 17
# dog = 18
# #   

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["segmentations"],
    classes=["banana", "stop sign"],
    max_samples=5000,
    dataset_dir="G:\Datasets\CoCo\\biclass"
)

session.dataset = dataset

input("Press Enter to continue...")

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    classes=["banana", "stop sign"],
    max_samples=1000,
    dataset_dir="G:\Datasets\CoCo\\biclass"
)

session.dataset = dataset

print(dataset.default_classes)

