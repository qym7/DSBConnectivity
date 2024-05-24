import os

import PIL
import torch

from .utils import download_file_from_google_drive, check_integrity
from .vision import VisionDataset


class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                ``identity`` (int): label for each person (data points with the same identity are the same person)
                ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                    righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)
            Defaults to ``attr``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "./"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        ["1ak57neUpo1hbikxozBCWTe_Vf3MmqnAT", "list_landmarks_align_celeba.txt"],
        ["1g9dqxOv-jrkR_1p9hhsHwRZqdLUNS6XG", "list_eval_partition.txt"],
        ["1raAP7l0kKaZg7W01Zk8DyAyN4B_E41TJ", "list_bbox_celeba.txt"],
        ["1zGgRizyR872PG1N4TzVuI7t1BsX4lqpq", "list_attr_celeba.txt"],
        ["1T_FfvbnT7NwqGYwF9-OB4ZBXaTK0hb4c", "img_align_celeba.zip"],
        ["1F5XjLVZ7PjTDybzUDV9pi6KUmf_9j_Nz", "identity_CelebA.txt"],
    ]

    def __init__(
        self,
        root,
        split="train",
        target_type="attr",
        transform=None,
        target_transform=None,
        download=False,
    ):
        import pandas

        super(CelebA, self).__init__(root)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]
        self.transform = transform
        self.target_transform = target_transform

        # if download:
        #     self.download()

        self.transform = transform
        self.target_transform = target_transform

        if split.lower() == "train":
            split = 0
        elif split.lower() == "valid":
            split = 1
        elif split.lower() == "test":
            split = 2
        else:
            raise ValueError(
                'Wrong split entered! Please use split="train" '
                'or split="valid" or split="test"'
            )

        with open(
            os.path.join(self.root, self.base_folder, "list_eval_partition.txt"), "r"
        ) as f:
            splits = pandas.read_csv(f, delim_whitespace=True, header=None, index_col=0)

        with open(
            os.path.join(self.root, self.base_folder, "identity_CelebA.txt"), "r"
        ) as f:
            self.identity = pandas.read_csv(
                f, delim_whitespace=True, header=None, index_col=0
            )

        with open(
            os.path.join(self.root, self.base_folder, "list_bbox_celeba.txt"), "r"
        ) as f:
            self.bbox = pandas.read_csv(f, delim_whitespace=True, header=1, index_col=0)

        with open(
            os.path.join(
                self.root, self.base_folder, "list_landmarks_align_celeba.txt"
            ),
            "r",
        ) as f:
            self.landmarks_align = pandas.read_csv(f, delim_whitespace=True, header=1)

        with open(
            os.path.join(self.root, self.base_folder, "list_attr_celeba.txt"), "r"
        ) as f:
            self.attr = pandas.read_csv(f, delim_whitespace=True, header=1)

        mask = splits[1] == split
        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(self.identity[mask].values)
        self.bbox = torch.as_tensor(self.bbox[mask].values)
        self.landmarks_align = torch.as_tensor(self.landmarks_align[mask].values)
        self.attr = torch.as_tensor(self.attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}

    def download(self):
        import zipfile

        for file_id, filename in self.file_list:
            fp = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fp):
                download_file_from_google_drive(
                    file_id, os.path.join(self.root, self.base_folder), filename
                )

        with zipfile.ZipFile(
            os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r"
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "img_align_celeba", self.filename[index]
            )
        )

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError('Target type "{}" is not recognized.'.format(t))
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
