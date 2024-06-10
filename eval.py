import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk

from models.moduleRCA import *
from utils.utils import dice_coefficient, iou_coefficient
from Options.TestOptions import TestOption
from scipy.ndimage import gaussian_filter, binary_erosion
from sklearn.metrics import confusion_matrix

class Eval():
    def __init__(self,
                 args,
                 test_ct,
                 test_hippo,
                 model):
        self.args = args
        self.model = model
        self.model_save_path = os.path.join(args.model_save_path, args.date, "model_parameters", f"{args.model}_{args.filename}.pt")
        self.test_ct = test_ct
        self.test_hippo = test_hippo
        self.save_path = f"D:\\HIPPO\\{args.date}\\test"

    def get_eval_metric(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()

        cm = confusion_matrix(pred, target)
        TP = cm[1][1]
        TN = cm[0][0]

        FP = cm[0][1]
        FN = cm[1][0]

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = TP / (TP + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        return f1_score, iou, precision, recall, accuracy

    # def remove_false_negative(self, target, pred):
    #     d, h, w = target.shape

    #     for i in range(d):
    #         for j in range(h):
    #             for z in range(w):
    #                 # target에서 0

    def load(self):
        model = self.model.to(self.args.device)
        model = torch.nn.DataParallel(model).to(self.args.device)
        self.model.load_state_dict(torch.load(self.model_save_path))
        print(f"Model Load Complete!")

    def get_volume(self, path):
        volume = sitk.ReadImage(path)
        volume = sitk.GetArrayFromImage(volume)

        volume = np.transpose(volume, (1,0,2))
        volume = np.rot90(volume, 2) 

        return volume
    
    def thresholding(self, volume, threshold):
        copy_volume = volume.copy()

        copy_volume[copy_volume > threshold] = 1
        copy_volume[copy_volume <= threshold] = 0

        return copy_volume

    def crop__volume(self, volume, crop_size):
        copy_volume = volume.copy()

        d, h, w = volume.shape
        
        start_z = d // 2
        start_x = h // 2
        start_y = w // 2

        cropped_volume = copy_volume[start_z - crop_size[0] // 2 : start_z + crop_size[0] // 2,
                                    start_x - crop_size[1] // 2 : start_x + crop_size[1] // 2,
                                    start_y - crop_size[2] // 2 : start_y + crop_size[2] // 2,]
        
        return cropped_volume

    # For CT volume function
    def __minmaxnormalize(self, volume):
        copy_volume = volume.copy()

        s = np.min(volume)
        b = np.max(volume)

        return (copy_volume - s) / (b - s)
    
    def __adjust__window(self, volume):
        copy_volume = volume.copy()

        min_window = self.args.min_window
        max_window = self.args.max_window

        copy_volume[copy_volume <= min_window] = min_window
        copy_volume[copy_volume >= max_window] = max_window

        return copy_volume
    
    def bone_processing(self, volume):
        copy_volume = volume.copy()

        copy_volume[copy_volume >= self.args.bone_threshold] = 0

        return copy_volume
    
    # For HIPPO volume function
    def __get_binary_volume(self, volume):
        copy_volume = volume.copy()

        copy_volume[copy_volume != 0] = 1
    
        return copy_volume
    
    def get_boundary_map(self, volume):
        filter_data = gaussian_filter(volume, self.args.gaussian_filter)
        threshold = self.args.filter_threshold

        binary_mask = filter_data > threshold

        eroded_mask = binary_erosion(binary_mask)
        boundary_map = binary_mask.astype(int) - eroded_mask.astype(int)

        return boundary_map

    def make_data(self, index):
        test_ct = self.test_ct[index]
        test_hippo = self.test_hippo[index]

        # load data
        ct = self.get_volume(test_ct)
        hippo = self.get_volume(test_hippo)

        # crop volume
        ct = self.crop__volume(ct, (self.args.depth_crop_size, 
                                    self.args.crop_size,
                                    self.args.crop_size))
        hippo = self.crop__volume(hippo, (self.args.depth_crop_size, 
                                    self.args.crop_size,
                                    self.args.crop_size))
        
        ct = self.__adjust__window(ct)
        ct = self.__minmaxnormalize(ct)
        ct = self.bone_processing(ct)
        ct = self.__minmaxnormalize(ct)

        hippo = self.__get_binary_volume(hippo)
        boundary = self.get_boundary_map(hippo)

        boundary = self.crop__volume(boundary, (self.args.depth_crop_size, self.args.crop_size, self.args.crop_size))

        return torch.from_numpy(ct).unsqueeze(0).unsqueeze(0), torch.from_numpy(hippo).unsqueeze(0).unsqueeze(0), torch.from_numpy(boundary).unsqueeze(0).unsqueeze(0)
    
    def gpu_to_cpu(self, volume, temp=None):
        if temp:
            return volume.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            return volume.squeeze(0).detach().cpu().numpy()
    # self.save_volume(ct, folder, model_folder, 'ct')
    def save_volume(self, volume, folder, model_folder, name, feature_folder=None):
        length = len(volume.shape)
        if length == 3:
            sitk.WriteImage(sitk.GetImageFromArray(volume), 
                            f"D:\\HIPPO\\{self.args.date}\\test\\{folder}\\{model_folder}\\{name}.nii.gz")

        elif length == 4:
            slice = volume.shape[0]
            path = f"D:\\HIPPO\\{self.args.date}\\test\\{folder}\\{model_folder}\\{feature_folder}"
            os.makedirs(f"{path}", exist_ok=True)

            for i in range(slice):
                sitk.WriteImage(sitk.GetImageFromArray(volume[i]),
                                os.path.join(path, f"{i+1}th_feature_map.nii.gz"))

    def evaluation(self):
        # load model
        self.load()

        # make data
        datasize = len(self.test_ct)
        print(f"test size : {datasize}")

        total_dice0, total_iou0, total_pre0, total_recall0, total_acc0 = 0,0,0,0,0
        total_dice1, total_iou1, total_pre1, total_recall1, total_acc1 = 0,0,0,0,0
        total_dice2, total_iou2, total_pre2, total_recall2, total_acc2 = 0,0,0,0,0
        total_dice3, total_iou3, total_pre3, total_recall3, total_acc3 = 0,0,0,0,0
        total_dice4, total_iou4, total_pre4, total_recall4, total_acc4 = 0,0,0,0,0
        total_dice5, total_iou5, total_pre5, total_recall5, total_acc5 = 0,0,0,0,0
        total_dice6, total_iou6, total_pre6, total_recall6, total_acc6 = 0,0,0,0,0
        total_dice7, total_iou7, total_pre7, total_recall7, total_acc7 = 0,0,0,0,0
        total_dice8, total_iou8, total_pre8, total_recall8, total_acc8 = 0,0,0,0,0
        total_dice9, total_iou9, total_pre9, total_recall9, total_acc9 = 0,0,0,0,0

        for i in range(datasize):
            ct, hippo, edge = self.make_data(i)
            folder = self.test_ct[i].split('\\')[3]

            model_folder = f"{self.args.model}_{self.args.filename}"                                                                 

            os.makedirs(f"D:\\HIPPO\\{self.args.date}\\test\\{folder}", exist_ok=True)
            os.makedirs(f"D:\\HIPPO\\{self.args.date}\\test\\{folder}\\{model_folder}", exist_ok=True)
            
            # predict
            if self.args.edge == 1:
                if self.args.module:
                    pred, edge_pred = self.model(ct.to(self.args.device).float())
                    
                else:
                    pred, edge_pred, \
                    x1, x2, x3, x4, x5, x1_d, x2_d, x3_d, x4_d, \
                    edge_x1_d, edge_x2_d, edge_x3_d, edge_x4_d = self.model(ct.to(self.args.device).float())

            else:
                pred, \
                x1, x2, x3, x4, x5, x1_d, x2_d, x3_d, x4_d = self.model(ct.to(self.args.device).float())
            
            # ct, hippo, edge => cpu && save
            ct = self.gpu_to_cpu(ct, 1)
            hippo = self.gpu_to_cpu(hippo, 1)
            edge = self.gpu_to_cpu(edge, 1)
            
            self.save_volume(ct, folder, model_folder, 'ct')
            self.save_volume(hippo, folder, model_folder, 'hippo')            
            self.save_volume(edge, folder, model_folder, 'edge') 
            
            # save original pred
            threshold = self.args.test_threshold
            pred = np.clip(self.gpu_to_cpu(pred, 1), 0, 1)
            self.save_volume(self.thresholding(pred, threshold), folder, model_folder, 'pred')

            if self.args.edge == 1:
                edge = self.gpu_to_cpu(edge_pred, 1)
                self.save_volume(edge_pred, folder, model_folder, 'edge_pred')
            print(f"folder {folder} save!")

            # save feature map
            ## encoder feature map
            #if self.args.save == 1:
                # self.save_volume(self.gpu_to_cpu(x1), folder, model_folder, '', 
                #                 feature_folder='encoder1')
                # self.save_volume(self.gpu_to_cpu(x2), folder, model_folder, '', 
                #                 feature_folder='encoder2')
                # self.save_volume(self.gpu_to_cpu(x3), folder, model_folder, '', 
                #                 feature_folder='encoder3')
                # self.save_volume(self.gpu_to_cpu(x4), folder, model_folder, '', 
                #                 feature_folder='encoder4')
                # self.save_volume(self.gpu_to_cpu(x5), folder, model_folder, '', 
                #                 feature_folder='encoder5')
                
                # ## seg decoder feature map
                # self.save_volume(self.gpu_to_cpu(x1_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder1')
                # self.save_volume(self.gpu_to_cpu(x2_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder2')
                # self.save_volume(self.gpu_to_cpu(x3_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder3')
                # self.save_volume(self.gpu_to_cpu(x4_d), folder, model_folder, '', 
                #                 feature_folder='segdecoder4')
    
                # ## edge decoder feature map
                # if self.args.edge:
                #     self.save_volume(self.gpu_to_cpu(edge_x1_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder1')
                #     self.save_volume(self.gpu_to_cpu(edge_x2_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder2')
                #     self.save_volume(self.gpu_to_cpu(edge_x3_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder3')
                #     self.save_volume(self.gpu_to_cpu(edge_x4_d), folder, model_folder, '', 
                #                     feature_folder='edgedecoder4')
    
                # ## Module feature map
                # if self.args.module:
                #     self.save_volume(self.gpu_to_cpu(module1_seg), folder, model_folder, '', 
                #                     feature_folder='module1_seg')
                #     self.save_volume(self.gpu_to_cpu(module1_edge), folder, model_folder, '', 
                #                     feature_folder='module1_edge')
                #     self.save_volume(self.gpu_to_cpu(module2_seg), folder, model_folder, '', 
                #                     feature_folder='module2_seg')
                #     self.save_volume(self.gpu_to_cpu(module2_edge), folder, model_folder, '', 
                #                     feature_folder='module2_edge')

            for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                dice, iou, pre, recall, acc = self.get_eval_metric(self.thresholding(pred, threshold), hippo)

                if threshold == 0:
                    total_dice0 += dice
                    total_iou0 += iou
                    total_pre0 += pre
                    total_recall0 += recall
                    total_acc0 += acc
                elif threshold == 0.1:
                    total_dice1 += dice
                    total_iou1 += iou  
                    total_pre1 += pre
                    total_recall1 += recall
                    total_acc1 += acc
                elif threshold == 0.2:
                    total_dice2 += dice
                    total_iou2 += iou 
                    total_pre2 += pre
                    total_recall2 += recall
                    total_acc2 += acc
                elif threshold == 0.3:
                    total_dice3 += dice
                    total_iou3 += iou 
                    total_pre3 += pre
                    total_recall3 += recall
                    total_acc3 += acc
                elif threshold == 0.4:
                    total_dice4 += dice
                    total_iou4 += iou 
                    total_pre4 += pre
                    total_recall4 += recall
                    total_acc4 += acc
                elif threshold == 0.5:
                    total_dice5 += dice
                    total_iou5 += iou 
                    total_pre5 += pre
                    total_recall5 += recall
                    total_acc5 += acc
                elif threshold == 0.6:
                    total_dice6 += dice
                    total_iou6 += iou 
                    total_pre6 += pre
                    total_recall6 += recall
                    total_acc6 += acc
                elif threshold == 0.7:
                    total_dice7 += dice
                    total_iou7 += iou 
                    total_pre7 += pre
                    total_recall7 += recall
                    total_acc7 += acc
                elif threshold == 0.8:
                    total_dice8 += dice
                    total_iou8 += iou 
                    total_pre8 += pre
                    total_recall8 += recall
                    total_acc8 += acc
                elif threshold == 0.9:
                    total_dice9 += dice
                    total_iou9 += iou 
                    total_pre9 += pre
                    total_recall9 += recall
                    total_acc9 += acc
                        
                print(f"{folder} {threshold}=>  Dice : {dice:>.3f} IOU : {iou:>.3f} Pre : {pre:>.3f} Recall : {recall:>.3f} ACC : {acc:>.3f}")
                #torch.cuda.empty_cache()
            print()
          
        print(f"Mean Dice 0.0 : {total_dice0/datasize:>.3f}")
        print(f"Mean IOU  0.0 : {total_iou0/datasize:>.3f}")
        print(f"Mean Precision 0.0 : {total_pre0/datasize:>.3f}")
        print(f"Mean Recall 0.0 : {total_recall0/datasize:>.3f}")
        print(f"Mean ACC 0.0 : {total_acc0/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.1 : {total_dice1/datasize:>.3f}")
        print(f"Mean IOU  0.1 : {total_iou1/datasize:>.3f}")
        print(f"Mean Precision 0.1 : {total_pre1/datasize:>.3f}")
        print(f"Mean Recall 0.1 : {total_recall1/datasize:>.3f}")
        print(f"Mean ACC 0.1 : {total_acc1/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.2 : {total_dice2/datasize:>.3f}")
        print(f"Mean IOU  0.2 : {total_iou2/datasize:>.3f}")
        print(f"Mean Precision 0.2 : {total_pre2/datasize:>.3f}")
        print(f"Mean Recall 0.2 : {total_recall2/datasize:>.3f}")
        print(f"Mean ACC 0.2 : {total_acc2/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.3 : {total_dice3/datasize:>.3f}")
        print(f"Mean IOU  0.3 : {total_iou3/datasize:>.3f}")
        print(f"Mean Precision 0.3 : {total_pre3/datasize:>.3f}")
        print(f"Mean Recall 0.3 : {total_recall3/datasize:>.3f}")
        print(f"Mean ACC 0.3 : {total_acc3/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.4 : {total_dice4/datasize:>.3f}")
        print(f"Mean IOU  0.4 : {total_iou4/datasize:>.3f}")
        print(f"Mean Precision 0.4 : {total_pre4/datasize:>.3f}")
        print(f"Mean Recall 0.4 : {total_recall4/datasize:>.3f}")
        print(f"Mean ACC 0.4 : {total_acc4/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.5 : {total_dice5/datasize:>.3f}")
        print(f"Mean IOU  0.5 : {total_iou5/datasize:>.3f}")
        print(f"Mean Precision 0.5 : {total_pre5/datasize:>.3f}")
        print(f"Mean Recall 0.5 : {total_recall5/datasize:>.3f}")
        print(f"Mean ACC 0.5 : {total_acc5/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.6 : {total_dice6/datasize:>.3f}")
        print(f"Mean IOU  0.6 : {total_iou6/datasize:>.3f}")
        print(f"Mean Precision 0.6 : {total_pre6/datasize:>.3f}")
        print(f"Mean Recall 0.6 : {total_recall6/datasize:>.3f}")
        print(f"Mean ACC 0.6 : {total_acc6/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.7 : {total_dice7/datasize:>.3f}")
        print(f"Mean IOU  0.7 : {total_iou7/datasize:>.3f}")
        print(f"Mean Precision 0.7 : {total_pre7/datasize:>.3f}")
        print(f"Mean Recall 0.7 : {total_recall7/datasize:>.3f}")
        print(f"Mean ACC 0.7 : {total_acc7/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.8 : {total_dice8/datasize:>.3f}")
        print(f"Mean IOU  0.8 : {total_iou8/datasize:>.3f}")
        print(f"Mean Precision 0.8 : {total_pre8/datasize:>.3f}")
        print(f"Mean Recall 0.8 : {total_recall8/datasize:>.3f}")
        print(f"Mean ACC 0.8 : {total_acc8/datasize:>.3f}")
        print()
        print(f"Mean Dice 0.9 : {total_dice9/datasize:>.3f}")
        print(f"Mean IOU  0.9 : {total_iou9/datasize:>.3f}")
        print(f"Mean Precision 0.9 : {total_pre9/datasize:>.3f}")
        print(f"Mean Recall 0.9 : {total_recall9/datasize:>.3f}")
        print(f"Mean ACC 0.9 : {total_acc9/datasize:>.3f}")
        print()



if __name__=='__main__':
    opt = TestOption()
    args = opt.parse()

    device = args.device
    print(f"Device : {device}")

    test_ct = pd.read_excel(f"D:\\HIPPO\\test_.xlsx")['CT']
    test_hippo = pd.read_excel(f"D:\\HIPPO\\test_.xlsx")['HIPPO']

    model = Model(1, 1).to(device)

    model = torch.nn.DataParallel(model).to(device)

    evaluator = Eval(args,
                     test_ct,
                     test_hippo,
                     model)
    
    evaluator.evaluation()