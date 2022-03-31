import torch
import torch.nn as nn
import os

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu

from datasets.data_manager import UPLDataManager

from .utils import plotLogitsMap, plotPRMap

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    # semi-supervised templates
    "SSOxfordPets": "a photo of a {}, a type of pet.",
    "SSOxfordFlowers": "a photo of a {}, a type of flower.",
    "SSFGVCAircraft": "a photo of a {}, a type of aircraft.",
    "SSDescribableTextures": "{} texture.",
    "SSEuroSAT": "a centered satellite photo of {}.",
    "SSStanfordCars": "a photo of a {}.",
    "SSFood101": "a photo of {}, a type of food.",
    "SSSUN397": "a photo of a {}.",
    "SSCaltech101": "a photo of a {}.",
    "SSUCF101": "a photo of a person doing {}.",
    "SSImageNet": "a photo of a {}.",
}




@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        dm = UPLDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        if self.cfg.DATALOADER.OPEN_SETTING:
            self.test_novel_loader = dm.test_novel_loader
            self.test_base_loader = dm.test_base_loader

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits, image_features, self.text_features
    
    @torch.no_grad()
    def test(self, split=None, trainer_list=None):
        """A generic testing pipeline."""
        # 如果是ensemble的需要添加tag 在实验记录中方便区分
        # if trainer_list is not None and "ENSEMBLE:{}".format(self.cfg.TRAINER.ENSEMBLE_NUM) not in self.online_logger.tags:
        #     self.online_logger.tags += ("ENSEMBLE:{}".format(self.cfg.TRAINER.ENSEMBLE_NUM),)

        # if self.cfg.DATASET.NAME not in self.online_logger.tags:
        #     self.online_logger.tags += ("DATASET INFERENCE:{}".format(self.cfg.DATASET.NAME),)

        self.set_model_mode("eval")
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME, 
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):
            results_id += 1
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w')
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w')

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="train":
            data_loader = self.train_loader_x
            print("Do evaluation on train set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        outputs_for_tsne = []
        label_for_tsne = []
        image_features_for_tsne = []
        text_features_for_tsne = []
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if trainer_list is None or len(trainer_list)==1:
                # 如果不是ensemble的测试
                output, image_features, text_features = self.model_inference(input)
                image_features_for_tsne.append(image_features)
                text_features_for_tsne.append(text_features)
            else:
                # ensemble的测试
                outputs = [t.model_inference(input)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            self.evaluator.process(output, label, self.per_image_txt_writer, self.per_class_txt_writer)
            outputs_for_tsne.append(output)
            label_for_tsne.append(label)
        results = self.evaluator.evaluate()
        if split in ['all', 'train', 'test', 'novel', 'base']:
            if len(outputs_for_tsne) != 0:
                outputs_for_tsne = torch.cat(outputs_for_tsne, dim=0)
                print('outputs_for_tsne', outputs_for_tsne.shape)
                label_for_tsne = torch.cat(label_for_tsne, dim=0)
                print('label_for_tsne', label_for_tsne.shape)
                image_features_for_tsne = torch.cat(image_features_for_tsne, dim=0)
                text_features_for_tsne = text_features_for_tsne[0]
                torch.save(image_features_for_tsne, os.path.join(save_path, '{}_v_features.pt'.format(split)))
                torch.save(image_features_for_tsne, os.path.join(save_path, '{}_targets.pt'.format(split)))
                torch.save(outputs_for_tsne, os.path.join(save_path, '{}_logits.pt'.format(split)))
                torch.save(text_features_for_tsne, os.path.join(save_path, '{}_l_logits.pt'.format(split)))
                # plotLogitsMap(outputs_for_tsne, label_for_tsne, os.path.join(save_path, '{}_map.png'.format(split)), self.cfg.DATASET.NAME+'_'+split)
                plotPRMap(outputs_for_tsne, label_for_tsne, os.path.join(save_path, '{}_PR.png'.format(split)), self.cfg.DATASET.NAME+'_'+split)
    
                
                # self.online_logger.log({"inputs": wandb.Image('./{}.png'.format(split)), "captions": wandb.Html(split)})
            # wandb.run.summary["{}_accuracy".format(split)] = results['accuracy']
        self.per_image_txt_writer.close()
        self.per_class_txt_writer.close()
        

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

