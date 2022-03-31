from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper
from dassl.data.transforms import build_transform
from dassl.data.samplers import build_sampler

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

def build_data_loader(
    cfg,
    sampler_type="RandomSampler",
    sampler=None,
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    tag=None
):
    # Build sampler
    if sampler_type is not None:
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_source,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=n_ins,
        )
    else:
        sampler = sampler

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    if tag is None:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )




    return data_loader

class UPLDataManager(DataManager):
    def __init__(self,
                cfg,
                custom_tfm_train=None,
                custom_tfm_test=None,
                dataset_wrapper=None):
        super().__init__(cfg, custom_tfm_train, custom_tfm_test, dataset_wrapper)

        
        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test
        
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        # save cfg 
        self.cfg = cfg
        self.tfm_train = tfm_train
        self.dataset_wrapper = dataset_wrapper

        if cfg.DATALOADER.OPEN_SETTING:
            test_novel_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.dataset.novel,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

            test_base_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=self.dataset.base,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

            self.test_novel_loader = test_novel_loader
            self.test_base_loader = test_base_loader
        
        try:
            if self.dataset.sstrain:
                # 除了dataset的source是不一样的，其他跟trian都是一样的
                train_loader_sstrain = build_data_loader(
                    cfg,
                    sampler_type="SequentialSampler",
                    data_source=self.dataset.sstrain,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=dataset_wrapper,
                    tag='sstrain' # 初始化的时候需要设置这个来保证所有样本的载入
                )
                self.train_loader_sstrain = train_loader_sstrain

                # Build train_loader_x
                train_loader_x = build_data_loader(
                    cfg,
                    sampler_type="SequentialSampler",
                    data_source=self.dataset.train_x,
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    tfm=tfm_test, # 这个是不训练的，所以要用测试的配置，
                    is_train=False,
                    dataset_wrapper=dataset_wrapper,
                    tag='sstrain'
                )
                self.train_loader_x = train_loader_x
        except:
            pass
        
    def update_ssdateloader(self, predict_label_dict, predict_conf_dict):
        """update the train_loader_sstrain to add labels

        Args:
            predict_label_dict ([dict]): [a dict {'imagepath': 'label'}]
        """
    

        sstrain = self.dataset.add_label(predict_label_dict, self.cfg.DATASET.NAME)
        print('sstrain', len(sstrain))
        
        
        # train_sampler = WeightedRandomSampler(weights, len(sstrain))
        train_loader_sstrain = build_data_loader(
            self.cfg,
            sampler_type="RandomSampler", 
            sampler=None,
            data_source=sstrain,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=1, # 每个类别n_ins个instance
            tfm=self.tfm_train,
            is_train=True,
            dataset_wrapper=self.dataset_wrapper,
        )
        self.train_loader_sstrain = train_loader_sstrain
        