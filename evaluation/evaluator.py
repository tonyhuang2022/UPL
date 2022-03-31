import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from dassl.evaluation.evaluator import Classification
from dassl.evaluation.build import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register()
class UPLClassification(Classification):

    def process(self, mo, gt, per_image_txt_writer, per_class_txt_writer):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        self.per_image_txt_writer = per_image_txt_writer
        self.per_class_txt_writer = per_class_txt_writer
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, (label, pred) in enumerate(zip(gt, mo)):
                label = label.item()
                label_name = self._lab2cname[label]                
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)
                # print(i, label_name, label, matches_i, pred.data.cpu().numpy().tolist())
                write_line = str(i) + "," + str(label_name) + "," + str(label) + "," + str(matches_i) + "," + str(pred.data.cpu().numpy().tolist())
                self.per_image_txt_writer.write(write_line+'\n')

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in self._lab2cname:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    "* class: {} ({})\t"
                    "total: {:,}\t"
                    "correct: {:,}\t"
                    "acc: {:.2f}%".format(
                        label, classname, total, correct, acc
                    )
                )
                write_line = "* class: {} ({}), total: {:,}, correct: {:,}, acc: {:.2f}% \n".format(label, classname, total, correct, acc)
                self.per_class_txt_writer.write(write_line)
            mean_acc = np.mean(accs)
            print("* average: {:.2f}%".format(mean_acc))

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print('Confusion matrix is saved to "{}"'.format(save_path))

        return results
