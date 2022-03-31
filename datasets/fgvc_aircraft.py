import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from .oxford_pets import OxfordPets
from .datasetbase import UPLDatasetBase


@DATASET_REGISTRY.register()
class FGVCAircraft(DatasetBase):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items


@DATASET_REGISTRY.register()
class SSFGVCAircraft(UPLDatasetBase):
    dataset_dir = 'fgvc_aircraft'
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        
        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}
        self.cname2lab = cname2lab
        
        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        sstrain = self.read_data_without_label(cname2lab, "images_variant_train.txt")
        num_shots = cfg.DATASET.NUM_SHOTS
        train = self.generate_fewshot_dataset(train, num_shots=-1)
        val = self.generate_fewshot_dataset(val, num_shots=-1)  
        super().__init__(train_x=train, val=val, test=test, sstrain=sstrain)
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def read_data_without_label(self, cname2lab, split_file, predict_label_dict=None):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        if predict_label_dict is None:
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    imname = line[0] + ".jpg"
                    classname = " ".join(line[1:])
                    impath = os.path.join(self.image_dir, imname)
                    label = cname2lab[classname]
                    item = Datum(impath=impath, label=-1, classname=None)
                    items.append(item)
        else:
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(" ")
                    imname = line[0] + ".jpg"
                    classname = " ".join(line[1:])
                    impath = os.path.join(self.image_dir, imname)
                    sub_impath = './data/' + impath.split('/data/')[1]
                    if sub_impath in predict_label_dict:
                        item = Datum(impath=impath, label=predict_label_dict[sub_impath], classname=self._lab2cname[predict_label_dict[sub_impath]])
                        items.append(item)
        return items
