import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

'''Confusion Matrix'''
class ConfusionMatrix(object):

    def __init__(self,num_classes:int,labels:list):
        self.matrix=np.zeros((num_classes,num_classes)) #初始化混淆矩阵，元素都为0
        self.num_classes=num_classes #类别数量
        self.labels=labels #类别标签
        self.PrecisionofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SensitivityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SpecificityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.F1_scoreofEachClass = [0.0 for cols in range(self.num_classes)]
        self.acc = 0.0


    def update(self,pred,label):
       if len(pred)>1:
            for p,t in zip(pred, label): #pred为预测结果，labels为真实标签
                self.matrix[int(p),int(t)] += 1 #根据预测结果和真实标签的值统计数量，在混淆矩阵相应的位置+1
       else:
            self.matrix[int(pred),int(label)] += 1

    def summary(self,File):
        #calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i] #混淆矩阵对角线的元素之和，也就是分类正确的数量
        self.acc = sum_TP/np.sum(self.matrix) #总体准确率
        print("the model accuracy is :{:.4f}".format(self.acc))
        File.write("the model accuracy is {:.4f}".format(self.acc)+"\n")

        #precision,recall,specificity
        table=PrettyTable() #创建一个表格
        table.field_names=["","Precision","Sensitivity","Specificity","F1-score"]
        for i in range(self.num_classes):
            TP=self.matrix[i,i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision=round(TP/(TP+FP),4) if TP+FP!=0 else 0.
            Sensitivity=round(TP/(TP+FN),4) if TP+FN!=0 else 0.
            Specificity=round(TN/(TN+FP),4) if TN+FP!=0 else 0.
            F1_score = round((2*Sensitivity*Precision)/(Sensitivity+Precision),4) if (Sensitivity!=0 and Specificity!=0) else 0

            self.PrecisionofEachClass[i]=Precision
            self.SensitivityofEachClass[i]=Sensitivity
            self.SpecificityofEachClass[i]=Specificity
            self.F1_scoreofEachClass[i] = F1_score

            table.add_row([self.labels[i],Precision,Sensitivity,Specificity,F1_score])
        print(table)
        File.write(str(table)+'\n')
        return self.acc

    def get_f1score(self):
        return self.F1_scoreofEachClass


'''ROC AUC'''
def Auc(pro_list, lab_list, classnum, File, task='validate'):
    pro_array = np.array(pro_list)
    lab_array = np.array(lab_list)
    
    # 创建一个PrettyTable表格
    table = PrettyTable()
    table.field_names = ["Class", "AUC"]
    roc_aucs = []

    plt.figure(figsize=(10, 5))

    # 画每个类别的ROC曲线
    for i in range(classnum):
        fpr, tpr, _ = roc_curve(lab_array == i, pro_array[:, i])
        auc_score = roc_auc_score(lab_array == i, pro_array[:, i])
        roc_aucs.append(auc_score)
        table.add_row([i, auc_score])
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
        plt.title(f'{task} ROC Curves')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

    # 画AUC的堆积图
    plt.subplot(1, 2, 2)
    plt.bar(range(classnum), roc_aucs, color='skyblue')
    plt.title('AUC Scores by Class')
    plt.xlabel('Class')
    plt.ylabel('AUC Score')

    # 显示图表
    plt.tight_layout()
    plt.savefig(f'./results/ISIC2018/{task} ROC-AUC Curves.jpg')

    # 写入文件和打印表格
    print(table)
    File.write(str(table) + '\n')
    print("The average AUC: {:.4f}".format(np.mean(roc_aucs)))
