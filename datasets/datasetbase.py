from dassl.data.datasets import DatasetBase
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json
from dassl.utils import listdir_nohidden

import random
import os


class UPLDatasetBase(DatasetBase):
    def __init__(self, train_x=None, train_u=None, val=None, test=None, novel=None, base=None, sstrain=None):
        super().__init__(train_x=train_x, val=val, test=test)
        self._novel = novel
        self._base = base
        self.sstrain = sstrain
        self._lab2cname_novel, _  = self.get_lab2cname(novel) 
        self._lab2cname_base, _ = self.get_lab2cname(base)

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, mode=None, ignore_labels=None, repeat=False):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            
            for label, items in tracker.items():
                # print(label, items[0].classname)
                if mode == 'train':
                    if ignore_labels:
                        if items[0].classname not in ignore_labels:
                            if len(items) >= num_shots:
                                sampled_items = random.sample(items, num_shots)
                            else:
                                if repeat:
                                    sampled_items = random.choices(items, k=num_shots)
                                else:
                                    sampled_items = items
                            dataset.extend(sampled_items)
                    else:
                        if len(items) >= num_shots:
                            sampled_items = random.sample(items, num_shots)
                        else:
                            if repeat:
                                sampled_items = random.choices(items, k=num_shots)
                            else:
                                sampled_items = items
                        dataset.extend(sampled_items)
                else:
                    if len(items) >= num_shots:
                        sampled_items = random.sample(items, num_shots)
                    else:
                        if repeat:
                            sampled_items = random.choices(items, k=num_shots)
                        else:
                            sampled_items = items
                    dataset.extend(sampled_items)
            output.append(dataset)
        if len(output) == 1:
            return output[0]

        return output
    
    def split_base_and_novel(self, test, ignore_labels):
        tracker = self.split_dataset_by_label(test)
        dataset_base = []
        dataset_novel = []
        for label, items in tracker.items():
            if items[0].classname not in ignore_labels:
                dataset_novel.extend(items)
            else:
                dataset_base.extend(items)
        return dataset_novel, dataset_base
    
    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        if data_source is not None:
            for item in data_source:
                container.add((item.label, item.classname))
            mapping = {label: classname for label, classname in container}
            labels = list(mapping.keys())
            labels.sort()
            classnames = [mapping[label] for label in labels]
            return mapping, classnames
        else:
            return None, None
    
    def read_split(self, filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=label, classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test
    
    def read_sstrain_data(self, filepath, path_prefix, predict_label_dict=None):
        
        def _convert(items):
            out = []
            for impath, _, _ in items:
                impath = os.path.join(path_prefix, impath)
                sub_impath = './data/' + impath.split('/data/')[1]
                if sub_impath in predict_label_dict:
                    item = Datum(impath=impath, label=predict_label_dict[sub_impath], classname=self._lab2cname[predict_label_dict[sub_impath]])
                    out.append(item)
            return out
        
        def _convert_no_label(items):
            out = []
            for impath, _, _ in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=-1, classname=None)
                out.append(item)
            return out
        
        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        if predict_label_dict is not None:
            train = _convert(split["train"])
        else:
            train = _convert_no_label(split["train"])
        return train
    
    # dtd会报错 需要单独处理，它里面有处理的函数，需要copy过来，详见原版的database

    def add_label(self, predict_label_dict, dataset_name):
        """add label when training for self-supervised learning

        Args:
            predict_label_dict ([dict]): [a dict {'imagepath': 'label'}]
        """
        # print(predict_label_dict, 'predict_label_dict')
        print(dataset_name)
        if dataset_name == 'SSFGVCAircraft':
            sstrain = self.read_data_without_label(self.cname2lab, "images_variant_train.txt", predict_label_dict)
        elif dataset_name == 'SSImageNet':
            sstrain = self.read_sstrain_data(self.train_x, predict_label_dict)
        else:
            sstrain = self.read_sstrain_data(self.split_path, self.image_dir, predict_label_dict)
        
        self.sstrain = sstrain
        return sstrain