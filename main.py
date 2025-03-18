from args import create_exp_dirs
from args import init_parser, init_sub_args
import torch
import random
import numpy as np
import copy

from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params, Trainer, init_optimizer, init_scheduler, CostumLoss
from utils.data_utils import trans_list
from utils.eval import score_dataset, combined_score_dataset
import yaml
import os
from models import *
from utils.tokenizer import Tokenizer
from utils.eval import eval

def main ():
    parser = init_parser()
    args = parser.parse_args()
    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)
 
    pretrained_model = args.branch == "SPARTA_H" or bool(vars(args).get('model_ckpt_dir'))
    recon_encoder = vars(args).get('recon_encoder_path', None)
    
    if args.branch == "SPARTA_H":
        args_c = copy.deepcopy(args)
        args_c.branch = "SPARTA_C"
        args_f = copy.deepcopy(args)
        args_f.branch = "SPARTA_F"
        dataset_F, loader_F = get_dataset_and_loader(args_f, trans_list=trans_list, only_test=(pretrained_model is not None))
        dataset, loader = get_dataset_and_loader(args_c, trans_list=trans_list, only_test=(pretrained_model is not None))
    else:
        dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained_model is not None))

    expand_ratio = 1
    extra_dim = 0
    if args.relative:
        expand_ratio = 2
        
    if model_args.token_config == "kps":
        input_dim = model_args.seg_len*2
    elif model_args.token_config == "2ds":
        input_dim = model_args.seg_len
    elif model_args.token_config == "st":
        input_dim = 2
    else:
        if args.traj:
            if args.relative:
                extra_dim = 4
            else:
                extra_dim = 2
        input_dim = args.num_kp*2
    
    if args.branch == "SPARTA_C":
        Transformer_model = SPARTA_C(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device, dropout=args.dropout)
        tokenizer = Tokenizer(args=args)
    elif args.branch == "SPARTA_F":
        Transformer_model = SPARTA_F(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device)
        tokenizer = Tokenizer(args=args)
    elif args.branch == "SPARTA_H":
        Transformer_model = SPARTA_H(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device, dropout=args.dropout)
        tokenizer_f = Tokenizer(args_f)  # Tokenizer for SPARTA_F logic
        tokenizer_c = Tokenizer(args_c)  # Tokenizer for SPARTA_C logic
   
    if pretrained_model == None:
        if not os.path.exists(args.model_save_dir):
        # Create the directory
            os.makedirs(args.model_save_dir)
        if recon_encoder != None:
            checkpoint = torch.load(args.recon_encoder_path)
            model_dict = Transformer_model.state_dict()
            encoder_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if 'encoder' in key:
                    encoder_state_dict[key] = value

            for name, param in encoder_state_dict.items():
                if name not in model_dict:
                    continue

                param = param.data
                model_dict[name].copy_(param)

            encoder_layers = Transformer_model.encoder.layers
            for l in encoder_layers:
                l.trainable = False
            print("Frozen Reconstruction Encoder Loaded!")
        
        
        arguments = vars(args)
        with open(args.model_save_dir + '/' + 'arguments.yaml', 'w') as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(args.sched, lr=args.model_lr, epochs=args.epochs)
        trainer = Trainer(model_args, Transformer_model, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                                scheduler_f=ae_scheduler_f)
        trained_model = trainer.train(checkpoint_filename='trans', args=args)
        
    else:
        if args.branch == "SPARTA_H":
            sparta_c_weights = torch.load(args.model_ckpt_C, map_location=args.device)
            sparta_f_weights = torch.load(args.model_ckpt_F, map_location=args.device)
            Transformer_model.CTD.load_state_dict(sparta_c_weights['state_dict'])
            Transformer_model.FTD.load_state_dict(sparta_f_weights['state_dict'])
        
        else:
            checkpoint = torch.load(args.model_ckpt_dir,  map_location=args.device)
            Transformer_model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully!')
        Transformer_model.to(args.device)
        loss_func = CostumLoss(model_args.loss, a=model_args.a, b=model_args.b, c=model_args.c, d=model_args.d)        
        
                    

        if args.branch == "SPARTA_H":
            print ("*********************SPARTA_C************************")
            eval_loss = eval (args_c, model_args, Transformer_model.CTD, tokenizer_c, loss_func, loader)
            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), dataset['test'].metadata, args=args_c)
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
            
            print ("*********************SPARTA_F************************")
            args.branch = "SPARTA_F"
            eval_loss_ = eval (args_f, model_args, Transformer_model.FTD, tokenizer_f, loss_func, loader_F)
            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss_), dataset_F['test'].metadata, args=args_f)
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
            
            print ("*********************SPARTA_H************************")
            args.branch = "SPARTA_H"
            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = combined_score_dataset(np.array(eval_loss), np.array(eval_loss_), dataset['test'].metadata, dataset_F['test'].metadata, args=args)
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
        else:
            eval_loss = eval (args, model_args, Transformer_model, tokenizer, loss_func,  loader)
            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), dataset['test'].metadata, args=args)
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
    
if __name__ == '__main__':
    main()
