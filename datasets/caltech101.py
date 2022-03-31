import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .datasetbase import UPLDatasetBase
from dassl.utils import read_json

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import random

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        super().__init__(train_x=train, val=val, test=test)

@DATASET_REGISTRY.register()
class OpensetCaltech101(UPLDatasetBase):

    dataset_dir = 'caltech-101'

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')
        
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(
                self.image_dir,
                ignored=IGNORED,
                new_cnames=NEW_CNAMES
            )
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        

        # if IGNORE_NUM <=0 use IGNORE_FILE
        ignore_label_num = int(cfg.DATASET.IGNORE_NUM)
        if ignore_label_num > 0:
            ignore_labels = [i for i in range(100)]
            ignore_labels = random.sample(ignore_labels, ignore_label_num)
        else:
            ignore_labels = []
            caltech101_ignore_file = open(cfg.DATASET.IGNORE_FILE, 'r')
            lines = caltech101_ignore_file.readlines()
            for line_id in range(0, len(lines)):
                class_name, is_ignore = lines[line_id].split(',')
                if int(is_ignore) == 1:
                    ignore_labels.append(class_name)
            

        num_shots = cfg.DATASET.NUM_SHOTS
        ignore_label_num = int(cfg.DATASET.IGNORE_NUM)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots, mode='train')
        val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4), mode='val')
        novel, base = self.split_base_and_novel(test, ignore_labels)
        self.novel = novel
        self.base = base

        super().__init__(train_x=train, val=val, test=test, novel=novel, base=base)

    def read_split(self, filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test



@DATASET_REGISTRY.register()
class SSCaltech101(UPLDatasetBase):
    dataset_dir = 'caltech-101'
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')
        
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(
                self.image_dir,
                ignored=IGNORED,
                new_cnames=NEW_CNAMES
            )
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        sstrain = self.read_sstrain_data(self.split_path, self.image_dir)
        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=-1)
        val = self.generate_fewshot_dataset(val, num_shots=-1)  
        super().__init__(train_x=train, val = val, test=test, sstrain=sstrain)
    
    