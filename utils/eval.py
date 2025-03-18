import os 
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
import csv
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
from sklearn.preprocessing import StandardScaler
import torch

target_fnr = 0.1

def combined_score_dataset(score_c, score_f, metadata_c, metadata_f, args=None):
    args.branch = "SPARTA_C"
    gt_arr_c, scores_arr_c = get_dataset_scores(score_c, metadata_c, args=args)
    scores_arr_c = smooth_scores(scores_arr_c)
    gt_np_c = np.concatenate(gt_arr_c)
    scores_np_c = np.concatenate(scores_arr_c)
    
    args.branch = "SPARTA_F"
    gt_arr_f, scores_arr_f = get_dataset_scores(score_f, metadata_f, args=args)
    scores_arr_f = smooth_scores(scores_arr_f)
    gt_np_f = np.concatenate(gt_arr_f)
    scores_np_f = np.concatenate(scores_arr_f)
    
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    scaled_c = scaler1.fit_transform(scores_np_c.reshape(-1, 1))
    scaled_f = scaler2.fit_transform(scores_np_f.reshape(-1, 1))
    combined_score = scaled_c + scaled_f
    auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr = score_auc(combined_score, gt_np_f)
    return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr

def score_dataset(score, metadata, args=None):
    gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr = score_auc(scores_np, gt_np)
    return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr

def get_dataset_scores(scores, metadata, args=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    clip_list = os.listdir(args.mask_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, args.mask_root, args)
        
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)
            if args.save_results:
                if not os.path.exists(args.save_results_dir):
                    os.mkdir(args.save_results_dir)
                with open(args.save_results_dir+clip.split(".")[0]+".csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(clip_gt, clip_score))

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc_roc = roc_auc_score(gt, scores_np)
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    
    auc_precision_recall = auc(recall, precision)

    fpr, tpr, threshold = roc_curve(gt, scores_np, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    idx_closest_fnr = np.argmin(np.abs(fnr - target_fnr))

    # Get the corresponding threshold and FPR values
    threshold_at_target_fnr = threshold[idx_closest_fnr]
    fpr_at_target_fnr = fpr [idx_closest_fnr]
    return auc_roc, auc_precision_recall, EER, eer_threshold, fpr_at_target_fnr, threshold_at_target_fnr


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):

    scene_id, clip_id = [int(i) for i in clip.split('.')[0].split('_')[:2]]
    if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
        return None, None
    
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)

    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf * -1
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        if args.branch == "SPARTA_C":
            clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores
        else:
            clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)+args.seg_len] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amax(clip_ppl_score_arr, axis=0)

    return clip_gt, clip_score

def eval (args, model_args, model, tokenizer, loss_func,  loader):
    eval_loss = []
    model.eval()
    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(loader['test'])):
            input_data, target_data = tokenizer.process_data(data_batch)
            if args.branch == "SPARTA_C":
                recon = model.forward(input_data, input_data)
                loss = loss_func.calculate(input_data, recon)
                eval_loss.extend(loss.cpu().numpy())
            elif args.branch == "SPARTA_F":
                pred = model.forward(input_data, target_data)
                loss = loss_func.calculate(target_data, pred)
                eval_loss.extend(loss.cpu().numpy())
    return eval_loss