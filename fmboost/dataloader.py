import os
import torch
import numpy as np
import torchvision
import webdataset as wds
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import ListConfig
from torch.utils.data import DataLoader
from PIL import Image
import random
import torchvision.transforms.functional as TF


from fmboost.helpers import instantiate_from_config
from fmboost.helpers import load_partial_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 val_batch_size: int = None,
                 train: dict = None,
                 validation: dict = None,
                 test: dict = None,
                 shuffle_validation: bool = False,
                 num_workers: int = 0,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        self.validation = validation
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.shuffle_validation = shuffle_validation

        self.dataset_configs = {}
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle_validation)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle_validation)

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)


""" WebDataset """


def identity(x):
    return x


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 tar_base,          # can be a list of paths or a single path
                 batch_size,
                 val_batch_size=None,
                 train=None,
                 validation=None,
                 test=None,
                 num_workers=4,
                 val_num_workers: int = None,
                 multinode=True,
                 remove_keys: list = None,          # list of keys to remove from the sample
                 ):
        super().__init__()
        if isinstance(tar_base, str):
            self.tar_base = tar_base
        elif isinstance(tar_base, ListConfig) or isinstance(tar_base, list):
            # check which tar_base exists
            for path in tar_base:
                if os.path.exists(path):
                    self.tar_base = path
                    break
            else:
                raise FileNotFoundError("Could not find a valid tarbase.")
        else:
            raise ValueError(f'Invalid tar_base type {type(tar_base)}')
        print(f'[WebDataModuleFromConfig] Setting tar base to {self.tar_base}')
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.rm_keys = remove_keys if remove_keys is not None else []

    def make_loader(self, dataset_config, train=True):
        image_transforms = []
        def lambda_fn(x):
            """Normalize to [-1, 1]"""
            return x * 2. - 1.
        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda_fn)])
        if 'image_transforms' in dataset_config:
            image_transforms.extend([instantiate_from_config(tt) for tt in dataset_config.image_transforms])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != 'identity' else identity
                for dkey in transforms_config}
        # this is crucial to set correct image key to get the transofrms applied correctly
        img_key = dataset_config.get('image_key', 'image.png')
        transform_dict.update({img_key: image_transforms})

        if 'dataset_transforms' in dataset_config:
            dataset_transforms = instantiate_from_config(dataset_config['dataset_transforms'])
        else:
            dataset_transforms = None

        if 'postprocess' in dataset_config:
            postprocess = instantiate_from_config(dataset_config['postprocess'])
        else:
            postprocess = None

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        if isinstance(dataset_config.shards, str):
            tars = os.path.join(self.tar_base, dataset_config.shards)
        elif isinstance(dataset_config.shards, list) or isinstance(dataset_config.shards, ListConfig):
            # decompose into lists of shards
            # Turn train-{000000..000002}.tar into ['train-000000.tar', 'train-000001.tar', 'train-000002.tar']
            tars = []
            for shard in dataset_config.shards:
                # Assume that the shard starts from 000000
                if '{' in shard:
                    start, end = shard.split('..')
                    start = start.split('{')[-1]
                    end = end.split('}')[0]
                    start = int(start)
                    end = int(end)
                    tars.extend([shard.replace(f'{{{start:06d}..{end:06d}}}', f'{i:06d}') for i in range(start, end+1)])
                else:
                    tars.append(shard)
            tars = [os.path.join(self.tar_base, t) for t in tars]
            # random shuffle the shards
            if shardshuffle:
                np.random.shuffle(tars)
        else:
            raise ValueError(f'Invalid shards type {type(dataset_config.shards)}')

        dset = wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'[WebDataModuleFromConfig] Loading {len(dset.pipeline[0].urls)} shards.')

        dset = (dset
                .decode('rgb', handler=wds.warn_and_continue)
                .map(self.filter_out_keys, handler=wds.warn_and_continue)
                .map_dict(**transform_dict, handler=wds.warn_and_continue)
                )

        # change name of image key to be consistent with other datasets
        renaming = dataset_config.get('rename', None)
        if renaming is not None:
            dset = dset.rename(**renaming)

        if dataset_transforms is not None:
            dset = dset.map(dataset_transforms)

        if postprocess is not None:
            dset = dset.map(postprocess)
        
        bs = self.batch_size if train else self.val_batch_size
        nw = self.num_workers if train else self.val_num_workers
        dset = dset.batched(bs, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=nw)

        return loader

    def filter_out_keys(self, sample):
        for key in self.rm_keys:
            sample.pop(key, None)
        return sample
    
    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)

class ResizeShortSide:
    """Resize PIL image so the shorter side equals `size`, keeping aspect ratio.

    Accepts either a single image or paired images (a, b) to be consistent with the
    paired transform pipeline.
    """
    def __init__(self, size, resample=Image.BICUBIC):
        self.size = int(size)
        self.resample = resample

    def _resize_one(self, img):
        w, h = img.size
        if min(w, h) == self.size:
            return img
        if w < h:
            new_w = self.size
            new_h = int(h * (self.size / w))
        else:
            new_h = self.size
            new_w = int(w * (self.size / h))
        return img.resize((new_w, new_h), resample=self.resample)

    def __call__(self, *imgs):
        # If only one image is provided, behave like a standard transform
        if len(imgs) == 1:
            return self._resize_one(imgs[0])
        # For paired inputs, resize both in the same way
        return tuple(self._resize_one(img) for img in imgs)


class PairedRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, a, b):
        # a and b are PIL Images with same size
        w, h = a.size
        th, tw = self.size, self.size
        if w == tw and h == th:
            return a, b
        if w < tw or h < th:
            # fallback to center crop after resize caller should ensure short side >= size
            return TF.center_crop(a, (th, tw)), TF.center_crop(b, (th, tw))
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return a.crop((x1, y1, x1 + tw, y1 + th)), b.crop((x1, y1, x1 + tw, y1 + th))


class PairedCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, a, b):
        return TF.center_crop(a, (self.size, self.size)), TF.center_crop(b, (self.size, self.size))


class PairedRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, a, b):
        if random.random() < self.p:
            return TF.hflip(a), TF.hflip(b)
        return a, b


class ToTensorNormalize:
    def __call__(self, a, b):
        a = TF.to_tensor(a)
        b = TF.to_tensor(b)
        # map to [-1, 1]
        a = a * 2.0 - 1.0
        b = b * 2.0 - 1.0
        return a, b


class PairedTransform:
    """Compose paired transforms that apply same spatial ops to both images."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, a, b):
        for t in self.transforms:
            # Some transforms (e.g., ResizeShortSide) accept either one or two args.
            try:
                a, b = t(a, b)
            except TypeError:
                a = t(a)
                b = t(b)
        return a, b


class PairedFolderDataset(torch.utils.data.Dataset):
    """Dataset for folders with structure: <root>/<split>/(degraded|clean)/images
    Example: root/train/degraded, root/train/clean, root/val/degraded, root/val/clean
    """
    def __init__(self, root, split='train', degraded_dir='input', clean_dir='target', transform=None):
        self.root = root
        self.split = split
        self.degraded_dir = os.path.join(root, split, degraded_dir)
        self.clean_dir = os.path.join(root, split, clean_dir)
        if not os.path.isdir(self.degraded_dir) or not os.path.isdir(self.clean_dir):
            raise FileNotFoundError(f"Expected '{degraded_dir}' and '{clean_dir}' under {os.path.join(root, split)}")

        def build_map(folder):
            m = {}
            for fn in os.listdir(folder):
                p = os.path.join(folder, fn)
                if not os.path.isfile(p):
                    continue
                name = os.path.splitext(fn)[0]
                m[name] = p
            return m

        inputs = build_map(self.degraded_dir)
        targets = build_map(self.clean_dir)
        common = sorted(set(inputs.keys()) & set(targets.keys()))
        self.items = [(inputs[k], targets[k]) for k in common]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        in_path, tgt_path = self.items[idx]
        in_img = Image.open(in_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')
        if self.transform is not None:
            in_img, tgt_img = self.transform(in_img, tgt_img)
        else:
            in_img = TF.to_tensor(in_img) * 2. - 1.
            tgt_img = TF.to_tensor(tgt_img) * 2. - 1.
        return {'image_degraded': in_img, 'image': tgt_img}


class PairedFolderDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size, image_size=256, val_batch_size=None, test_batch_size=None,
                 num_workers=4, seed=42, random_flip=True, pin_memory=True, persistent_workers=True,
                 prefetch_factor=2, degraded_dir='input', clean_dir='target'):
        super().__init__()
        # Expand '~' to absolute user home to avoid literal-tilde paths
        self.root = os.path.expanduser(root)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else self.val_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.seed = seed
        self.random_flip = random_flip
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir

        # placeholders to avoid attribute errors before setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # build transforms
        train_transforms = [ResizeShortSide(self.image_size), PairedRandomCrop(self.image_size)]
        if self.random_flip:
            train_transforms.append(PairedRandomHorizontalFlip())
        train_transforms.append(ToTensorNormalize())
        train_transform = PairedTransform(train_transforms)

        val_transforms = [ResizeShortSide(self.image_size), PairedCenterCrop(self.image_size), ToTensorNormalize()]
        val_transform = PairedTransform(val_transforms)

        self.train_dataset = PairedFolderDataset(
            self.root, split='train', degraded_dir=self.degraded_dir, clean_dir=self.clean_dir, transform=train_transform
        )
        val_degraded = os.path.join(self.root, 'val', self.degraded_dir)
        val_clean = os.path.join(self.root, 'val', self.clean_dir)
        if os.path.isdir(val_degraded) and os.path.isdir(val_clean):
            self.val_dataset = PairedFolderDataset(
                self.root, split='val', degraded_dir=self.degraded_dir, clean_dir=self.clean_dir, transform=val_transform
            )
        else:
            self.val_dataset = None

        test_degraded = os.path.join(self.root, 'test', self.degraded_dir)
        test_clean = os.path.join(self.root, 'test', self.clean_dir)
        if os.path.isdir(test_degraded) and os.path.isdir(test_clean):
            self.test_dataset = PairedFolderDataset(
                self.root, split='test', degraded_dir=self.degraded_dir, clean_dir=self.clean_dir, transform=val_transform
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dict_collation_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dict_collation_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dict_collation_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )


# -----------------------------------------------------------------------------
# Simple single-dataset loader (gt/degrade folders) for image restoration.
# This is a placeholder for later wiring in configs; not used by default.
# -----------------------------------------------------------------------------


class SingleTaskMixedDataset(torch.utils.data.Dataset):
    """Dataset for a single restoration task with folder structure:

    root/
      train/
        gt/*.png|jpg
        degrade/*.png|jpg

    It outputs tensors in [-1, 1] with keys: adap (degraded) and gt (clean).
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 256, random_flip: bool = True):
        self.root = root
        self.split = split
        self.image_size = int(image_size)
        self.random_flip = random_flip

        self.gt_dir = os.path.join(root, split, "gt")
        self.deg_dir = os.path.join(root, split, "degrade")
        if not (os.path.isdir(self.gt_dir) and os.path.isdir(self.deg_dir)):
            raise FileNotFoundError(f"Expected gt/degrade folders under {os.path.join(root, split)}")

        self.items = self._build_items()
        self.resize = ResizeShortSide(self.image_size)
        self.to_tensor = ToTensorNormalize()

    def _build_items(self):
        items = []
        for fn in os.listdir(self.gt_dir):
            name, ext = os.path.splitext(fn)
            gt_path = os.path.join(self.gt_dir, fn)
            deg_path = os.path.join(self.deg_dir, name + ext)
            if os.path.isfile(gt_path) and os.path.isfile(deg_path):
                items.append((deg_path, gt_path))
        if len(items) == 0:
            raise FileNotFoundError(f"No paired files found in {self.gt_dir} and {self.deg_dir}")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        deg_path, gt_path = self.items[idx]
        deg = Image.open(deg_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        # Resize short side to target, keep aspect, then center-crop to square.
        deg, gt = self.resize(deg), self.resize(gt)
        deg = TF.center_crop(deg, (self.image_size, self.image_size))
        gt = TF.center_crop(gt, (self.image_size, self.image_size))

        if self.random_flip and random.random() < 0.5:
            deg, gt = TF.hflip(deg), TF.hflip(gt)

        deg, gt = self.to_tensor(deg), self.to_tensor(gt)  # map to [-1, 1]
        # Return keys compatible with Trainer (image/image_degraded) and keep legacy aliases.
        return {
            "image": gt,
            "image_degraded": deg,
            "gt": gt,            # legacy alias
            "adap": deg,          # legacy alias
            "A_paths": deg_path,
            "B_paths": gt_path,
        }


class SingleTaskDataModule(pl.LightningDataModule):
    """Placeholder DataModule for single-task mixed dataset (train-only by default).

    Not wired into configs yet; kept for future use.
    """

    def __init__(self, root: str, batch_size: int = 2, image_size: int = 256, num_workers: int = 4,
                 val_batch_size: int = None, random_flip: bool = True):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.random_flip = random_flip

    def setup(self, stage=None):
        self.train_dataset = SingleTaskMixedDataset(
            root=self.root, split="train", image_size=self.image_size, random_flip=self.random_flip
        )
        # If a val split exists, instantiate; otherwise leave None.
        val_dir = os.path.join(self.root, "val")
        if os.path.isdir(val_dir):
            self.val_dataset = SingleTaskMixedDataset(
                root=self.root, split="val", image_size=self.image_size, random_flip=False
            )
        else:
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dict_collation_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            # No validation split available; return empty list to skip validation.
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dict_collation_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/faces_v0.yaml")
    datamod = WebDataModuleFromConfig(**config["data"]["params"])
    # from pudb import set_trace; set_trace()
    dataloader = datamod.train_dataloader()

    for i,batch in enumerate(dataloader):
        print(batch.keys())
        print(batch['image'].shape)
        print(f"Batch number: {i}")
        break
    print("end")
