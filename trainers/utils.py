from tkinter.tix import Tree
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import torch
from sklearn.metrics import precision_recall_curve


def plotLogitsMap(outputs, label, save_path, fig_title, max_lines=1000):
    fig, ax = plt.subplots(figsize=(5, 200))
    Softmax = torch.nn.Softmax(dim=1)
    output_m = Softmax(outputs)
    output_m = outputs.cpu().detach().numpy()
    
    pred = outputs.max(1)[1]
    matches = pred.eq(label).float()
    output_m = np.sort(output_m)
    output_m = output_m[:,::-1] # 从大到小排序
    output_m = output_m[:,:5] # 取前五个
    output_m_index = output_m[:,0].argsort()
    output_m = output_m[output_m_index]
    output_m = output_m[::-1,:] # 按第一列从大到小排序
    matches = matches[output_m_index] 
    matches = torch.flip(matches, dims=[0])
    matches = matches.cpu().detach().numpy()


    if len(matches) > max_lines:
        gap = int(len(matches) / 1000)
        index = np.arange(0, gap*1000, gap, int)
        output_m = output_m[index]
        matches = matches[index]
    print(save_path)
    matches = matches.tolist()


    im = ax.imshow(output_m, aspect='auto')
    ax.set_yticks(np.arange(output_m.shape[0]), labels=matches)
    for i, label in enumerate(ax.get_yticklabels()):
        if (int(matches[i])==0):
            label.set_color('red')
        elif (int(matches[i])==1):
            label.set_color('green')
            
    for i in range(output_m.shape[0]):
        for j in range(output_m.shape[1]):
            text = ax.text(j, i, str(round(output_m[i, j],2)),
                        ha="center", va="center", color="w")
    plt.title(fig_title)
    plt.savefig(save_path)
    plt.close()

def plotPRMap(outputs, label, save_path, fig_title):
    plt.figure(figsize=(15,15))
    plt.title('{} Precision/Recall Curve'.format(fig_title))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    output_m = outputs.cpu().detach().numpy()
    pred = outputs.max(1)[1]
    matches = pred.eq(label).float()
    output_m = np.sort(output_m)
    output_m = output_m[:,::-1] # 从大到小排序
    output_m = output_m[:,:5] # 取前五个
    output_m_index = output_m[:,0].argsort()
    output_m = output_m[output_m_index]
    output_m = output_m[::-1,:] # 按第一列从大到小排序
    matches = matches[output_m_index] 
    matches = torch.flip(matches, dims=[0])
    matches = matches.cpu().detach().numpy()
    # print(output_m[:,0].shape, matches.shape)
    precision, recall, thresholds = precision_recall_curve(matches, output_m[:,0])
    # print(precision)
    # print(recall)
    # print(thresholds, len(thresholds))
    plt.plot(recall, precision)

    step = 0
    for a, b, text in zip(recall, precision, thresholds):
        # if float(text) % 0.05 == 0:
        if step % 40 == 0:
            plt.text(a, b, text, ha='center', va='bottom', fontsize=10, color='blue')
            plt.plot(a, b, marker='o', color='red')
        step += 1
    plt.grid(ls='--')
    plt.savefig(save_path)
    plt.close()
 
def select_top_k_similarity_per_class(outputs, img_paths, K=1, image_features=None, is_softmax=True):
    # print(outputs.shape)
    if is_softmax:
        outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    from tqdm import tqdm
    for id in tqdm(list(set(ids.tolist()))): # 标签去重
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径

        if image_features is not None:
            img_features = image_features[index]
            if K >= 0:
                for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
            else:
                for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            if K >= 0:
                for img_path, conf in zip(img_paths_class[:K], conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
            else:
                for img_path, conf in zip(img_paths_class, conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

def select_by_conf(outputs, img_paths, K=1, conf_threshold=None, is_softmax=True):
    # print(outputs.shape)
    if is_softmax:
        outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签


    predict_label_dict = {}
    predict_conf_dict = {}
    from tqdm import tqdm
    for id in tqdm(list(set(ids.tolist()))): # 标签去重
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径

        for img_path, conf in zip(img_paths_class, conf_class):
            if conf > conf_threshold:
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

def select_top_k_similarity(outputs, img_paths, K=1, image_features=None, repeat=False):
    # print(outputs.shape)
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    # output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    conf_class = output_m_max[output_m_max_id] # 置信度
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    if image_features is not None:
        img_features = image_features
        if K >= 0:
            for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class):
                predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                predict_label_dict[img_path] = [id, img_feature, conf, logit]
    else:
        if K >= 0:
            for img_path, conf, id in zip(img_paths[:K], conf_class[:K], ids[:K]):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
        else:
            for img_path, conf, id in zip(img_paths, conf_class, ids):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict


def select_top_by_value(outputs, img_paths, conf_threshold=0.95, image_features=None, repeat=False):
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    # output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    conf_class = output_m_max[output_m_max_id] # 置信度
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    if image_features is not None:
        img_features = image_features
        for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
            if conf > conf_threshold:
                predict_label_dict[img_path] = [id, img_feature, conf, logit]  
    else:
        for img_path, id, conf in zip(img_paths, ids, conf_class):
            if conf > conf_threshold:
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict


def caculate_noise_rate(predict_label_dict, train_loader, trainer, sample_level=False):
    gt_label_dict = {}
    for batch_idx, batch in enumerate(train_loader):
        input, label, impath = trainer.parse_batch_test_with_impath(batch)
        for l, ip in zip(label, impath):
            if '/data/' in ip:
                ip = './data/' + ip.split('/data/')[1]
            gt_label_dict[ip] = l
    
    # print('gt_label_dict', len(gt_label_dict))
    # print('gt_label_dict', gt_label_dict)
    total = 0
    correct = 0
    for item in predict_label_dict:
        if '/data/' in item:
            item = './data/' + item.split('/data/')[1]
        if gt_label_dict[item] == predict_label_dict[item]:
            correct += 1
        total += 1
    print('Acc Rate {:.4f}'.format(correct/total))


def caculate_noise_rate_analyze(predict_label_dict, train_loader, trainer, sample_level=False):
    gt_label_dict = {}
    for batch_idx, batch in enumerate(train_loader):
        input, label, impath = trainer.parse_batch_test_with_impath(batch)
        for l, ip in zip(label, impath):
            ip = './data/' + ip.split('/data/')[1]
            gt_label_dict[ip] = l
    total = 0
    correct = 0
    for item in predict_label_dict:
        if gt_label_dict[item] == predict_label_dict[item][0]:
            correct += 1
            if sample_level is True:
                print(gt_label_dict[item], 1)
        total += 1
    print('Acc Rate {:.4f}'.format(correct/total))


def save_outputs(train_loader, trainer, predict_label_dict, dataset_name, text_features, backbone_name=None):
    backbone_name = backbone_name.replace('/', '-')
    gt_pred_label_dict = {}
    for batch_idx, batch in enumerate(train_loader):
        input, label, impath = trainer.parse_batch_test_with_impath(batch)
        for l, ip in zip(label, impath):
            l = l.item()
            ip = './data/' + ip.split('/data/')[1]
            if l not in gt_pred_label_dict:
                gt_pred_label_dict[l] = []
                pred_label = predict_label_dict[ip][0]
                pred_v_feature = predict_label_dict[ip][1]

                conf = predict_label_dict[ip][2]
                logits = predict_label_dict[ip][3]
                gt_pred_label_dict[l].append([ip, pred_label, pred_v_feature, conf, logits])
            else:
                pred_label = predict_label_dict[ip][0]
                pred_v_feature = predict_label_dict[ip][1]
                conf = predict_label_dict[ip][2]
                logits = predict_label_dict[ip][3]
                gt_pred_label_dict[l].append([ip, pred_label, pred_v_feature, conf, logits])
    
    idx = 0
    v_distance_dict = {}
    v_features = []
    logits_list = []
    for label in gt_pred_label_dict:
        avg_feature = None
        for item in gt_pred_label_dict[label]:
            impath, pred_label, pred_v_feature = item[0], item[1], item[2],
            if avg_feature is None:
                avg_feature = pred_v_feature.clone()
            else:
                avg_feature += pred_v_feature.clone()
        avg_feature /= len(gt_pred_label_dict[label]) # class center
        v_distance_dict_per_class = {}
        for item in gt_pred_label_dict[label]:
            impath, pred_label, pred_v_feature, conf, logits = item[0], item[1], item[2], item[3], item[4]
            v_features.append(pred_v_feature)
            logits_list.append(logits)
            v_dis = torch.dist(avg_feature, pred_v_feature, p=2)
            v_distance_dict_per_class[impath] = [idx, v_dis.item(), conf.item(), pred_label] # id, visual distance, confidence, predicted label 
            idx += 1
        v_distance_dict[label] = v_distance_dict_per_class

    v_features = torch.vstack(v_features)
    logits_tensor = torch.vstack(logits_list)

    if not os.path.exists('./analyze_results/{}/'.format(backbone_name)):
        os.makedirs('./analyze_results/{}/'.format(backbone_name))
    
    torch.save(v_features, './analyze_results/{}/{}_v_feature.pt'.format(backbone_name, dataset_name))
    torch.save(text_features, './analyze_results/{}/{}_l_feature.pt'.format(backbone_name, dataset_name))
    torch.save(logits_tensor, './analyze_results/{}/{}_logits.pt'.format(backbone_name, dataset_name))
    
    
    with open("./analyze_results/{}/{}.json".format(backbone_name, dataset_name), "w") as outfile:
        json.dump(v_distance_dict, outfile)


def select_top_k_similarity_per_class_with_high_conf(outputs, img_paths, K=1, image_features=None, repeat=False):
    # print(outputs.shape)
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签


    # 计算类别平均置信度
    class_avg_conf = {}
    for id in list(set(ids.tolist())): # 标签去重
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        class_avg_conf[id] = conf_class.sum() / conf_class.size
        # print(class_avg_conf[id])
    
    selected_ids = sorted(class_avg_conf.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:int(0.8*len(class_avg_conf))]
    remain_ids = sorted(class_avg_conf.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[int(0.8*len(class_avg_conf)):]

    selected_ids = [id[0] for id in selected_ids]
    remain_ids = [id[0] for id in remain_ids]

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}

    # selected_ids.append(0)

    for id in selected_ids: # 标签去重
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径

        if image_features is not None:
            img_features = image_features[index]
            if K >= 0:
                for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class[:K]):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
            else:
                for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            if K >= 0:
                for img_path, conf in zip(img_paths_class[:K], conf_class):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
            else:
                for img_path, conf in zip(img_paths_class, conf_class):
                    img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict, remain_ids, selected_ids


def select_top_k_similarity_per_class_with_low_conf(outputs, img_paths, conf_threshold, remain_ids, selected_ids, K=2):
    # print(outputs.shape)
    outputs = torch.nn.Softmax(dim=1)(outputs)
    remain_ids_list = remain_ids
    remain_ids = np.sort(np.array(remain_ids).astype(np.int))
    remain_logits = -100*torch.ones(outputs.shape).half().cuda()
    remain_logits[:, remain_ids] = outputs[:, remain_ids] * 5
    remain_logits = torch.nn.Softmax(dim=1)(remain_logits.float())
    outputs = remain_logits


    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签


    predict_label_dict = {}
    predict_conf_dict = {}
    no_sample_ids = []

    for id in remain_ids_list: # 标签去重
        # print(id)
        is_id_have_sample = False
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径

        if K >= 0:
            for img_path, conf in zip(img_paths_class[:K], conf_class[:K]):
                print(conf)
                if conf > 0.4:
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
                    is_id_have_sample = True
        else:
            for img_path, conf in zip(img_paths_class, conf_class):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
        # print(is_id_have_sample)
        if is_id_have_sample is False:
            no_sample_ids.append(id)

    print(no_sample_ids)
    return predict_label_dict, predict_conf_dict, no_sample_ids

def select_top_k_similarity_per_class_no_smaple(outputs, img_paths, no_sample_ids, K=16):
    # print(outputs.shape)
    outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签


    predict_label_dict = {}
    predict_conf_dict = {}

    for id in no_sample_ids: # 标签去重
        print(id)
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径

        if K >= 0:
            for img_path, conf in zip(img_paths_class[:K], conf_class[:K]):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
        else:
            for img_path, conf in zip(img_paths_class, conf_class):
                predict_label_dict[img_path] = id
                predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict


        