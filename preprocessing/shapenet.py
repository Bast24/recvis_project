"""Written using code parts from https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/dataset/dataset_shapenet.py"""

import os
import json
import copy

import numpy as np
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import preprocessing.pointcloud_processor as pointcloud_processor


class ShapeNet(data.Dataset):
    """ShapeNet (v1) dataset """

    SHAPENET13 = [
        "airplane",
        "bench",
        "cabinet",
        "car",
        "chair",
        "display",
        "lamp",
        "loudspeaker",
        "rifle",
        "sofa",
        "table",
        "telephone",
        "vessel",
    ]
    POINTCLOUD_PATH = "data/ShapeNetV1PointCloud"
    IMAGE_PATH = "data/ShapeNetV1Renderings"
    TAXONOMY_PATH = "data/taxonomy.json"
    CACHE_PATH = "data/cache"
    IDX_IMAGE_VAL = 0  # Hold out the first single-view to evaluate
    NUM_IMAGE_PER_OBJECT = 24  # There is a total of 25 views per object
    TRAIN_VAL_RATIO = 0.8

    def __init__(
        self, train=True, normalization="UnitBall", num_sample=2500, class_choice=None,
    ):
        # Chosen classes (list of names)
        if class_choice is None:
            class_choice = ShapeNet.SHAPENET13
        self.train = train
        # Number of sample per object
        self.num_sample = num_sample
        # Normalization of the point cloud
        self.normalization = normalization
        # Construct id to string and string to id class mappings, store chosen classes
        self.id2names, self.names2id, self.classes = self.load_taxonomy(class_choice)
        # Preprocess and cache
        self.data_metadata = []
        self.data_points = None
        self.datapath = []
        self.dataset_name = f"train_{normalization}_" + "_".join(class_choice)
        self.preprocess(normalization, class_choice)

    @classmethod
    def load_taxonomy(cls, class_choice):
        """Return the id to names and names to id string-string dicts"""
        assert os.path.exists(
            ShapeNet.TAXONOMY_PATH
        ), "Missing taxonomy. Please use the scripts to download the dataset."

        # Load all classes
        all_classes = [x for x in next(os.walk(cls.POINTCLOUD_PATH))[1]]

        id2names = {}
        names2id = {}
        with open(cls.TAXONOMY_PATH, "r") as f:
            taxonomy = json.load(f)
            for dict_class in taxonomy:
                if dict_class["synsetId"] in all_classes:
                    name = dict_class["name"].split(sep=",")[0]
                    id2names[dict_class["synsetId"]] = name
                    names2id[name] = dict_class["synsetId"]
        # Filter chosen classes
        classes = [x for x in all_classes if id2names[x] in class_choice]
        return id2names, names2id, classes

    def normalize(self, points):
        if self.normalization == "UnitBall":
            return pointcloud_processor.Normalization.normalize_unitL2ball_functional(
                points
            )
        elif self.normalization == "BoundingBox":
            return pointcloud_processor.Normalization.normalize_bounding_box_functional(
                points
            )
        else:
            return pointcloud_processor.Normalization.identity_functional(points)

    def preprocess(self, normalization, class_choice):
        # Init cache to store the preprocessed data
        self.dataset_path = os.path.join(ShapeNet.CACHE_PATH, self.dataset_name)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # Compile list of pointcloud path by selected category
        for category in self.classes:
            dir_pointcloud = os.path.join(ShapeNet.POINTCLOUD_PATH, category)
            dir_image = os.path.join(ShapeNet.IMAGE_PATH, category)
            list_pointcloud = sorted(os.listdir(dir_pointcloud))
            idx_split = int(len(list_pointcloud) * ShapeNet.TRAIN_VAL_RATIO)
            if self.train:
                list_pointcloud = list_pointcloud[:idx_split]
            else:
                list_pointcloud = list_pointcloud[idx_split:]

            if list_pointcloud:
                for pointcloud in list_pointcloud:
                    pointcloud_path = os.path.join(dir_pointcloud, pointcloud)
                    image_path = os.path.join(
                        dir_image, pointcloud.split(".")[0], "rendering"
                    )
                    if os.path.exists(image_path):
                        self.datapath.append(
                            (pointcloud_path, image_path, pointcloud, category)
                        )
                    else:
                        print(f"Rendering not found : {image_path}")

        pkl_fname = os.path.join(self.dataset_path, "info.pkl")
        points_fname = os.path.join(self.dataset_path, "points.pth")

        if os.path.exists(pkl_fname):
            # Reload dataset
            print(f"Reload dataset : {self.dataset_name}")
            with open(pkl_fname, "rb") as fp:
                self.data_metadata = pickle.load(fp)
            self.data_points = torch.load(points_fname)
        else:
            # Preprocess dataset and put in cache for future fast reload
            print("Preprocess dataset...")
            points_list = []
            for pointcloud_path, image_path, pointcloud, category in tqdm(
                self.datapath
            ):
                points = np.load(pointcloud_path)
                points = torch.from_numpy(points).float()
                points[:, :3] = self.normalize(points[:, :3])
                points = points.unsqueeze(0)
                points_list.append(points)
                metadata_dict = {
                    "pointcloud_path": pointcloud_path,
                    "image_path": image_path,
                    "name": pointcloud,
                    "category": category,
                }
                self.data_metadata.append(metadata_dict)

            self.data_points = torch.cat(points_list, 0)

            # Save in cache
            with open(pkl_fname, "wb") as fp:  # Pickling
                pickle.dump(self.data_metadata, fp)
            torch.save(self.data_points, points_fname)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        return_dict = copy.deepcopy(self.data_metadata[index])
        # Point processing
        points = self.data_points[index]
        points = points.clone()
        if self.num_sample:
            choice = np.random.choice(points.size(0), self.num_sample, replace=True)
            points = points[choice, :]
        return_dict["points"] = points[:, :3].contiguous()

        # Image processing
        if self.train:
            N = np.random.randint(1, ShapeNet.NUM_IMAGE_PER_OBJECT)
            img = Image.open(os.path.join(return_dict["image_path"], f"{N:02}.png"))
            img = self.train_transform(img)  # random crop
        else:
            N = ShapeNet.IDX_IMAGE_VAL
            img = Image.open(os.path.join(return_dict["image_path"], f"{N:02}.png"))
            img = self.val_transform(img)  # center crop
        img = self.common_transform(img)  # scale
        img = img[:3, :, :]
        return_dict["image"] = img
        return return_dict

    def train_transform(self, img):
        trf = transforms.Compose(
            [transforms.RandomCrop(127), transforms.RandomHorizontalFlip(),]
        )
        return trf(img)

    def val_transform(self, img):
        trf = transforms.Compose([transforms.CenterCrop(127),])
        return trf(img)

    def common_transform(self, img):
        trf = transforms.Compose(
            [transforms.Resize(size=224, interpolation=2), transforms.ToTensor(),]
        )
        return trf(img)


if __name__ == "__main__":
    s = ShapeNet(class_choice=["airplane"], train=True)

