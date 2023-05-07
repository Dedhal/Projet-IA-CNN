import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("cifar100", split="train", dataset_dir="G:\Datasets\Cifar100")

session = fo.launch_app(dataset)
