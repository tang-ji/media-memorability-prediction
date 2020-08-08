import numpy as np

class Metric:
    def __init__(self, prediction_path, labels=["negative", "Unknown", "positive"]):
        [self.Y, self.Y_pred] = np.load(prediction_path)
        self.threshold_positive, self.threshold_negative = 0.5, -0.5
        self.positive_precision = sum(self.Y[self.Y_pred>self.threshold_positive] > 0)/len(self.Y[self.Y_pred>self.threshold_positive])
        self.negative_precision = sum(self.Y[self.Y_pred<self.threshold_negative] < 0)/len(self.Y[self.Y_pred<self.threshold_negative])
        self.labels = labels
        
    def get_label(self, v):
        if v >= self.threshold_positive:
            return self.labels[2]
        elif v <= self.threshold_negative:
            return self.labels[0]
        else:
            return self.labels[1]
        
    def show_metrics(self):
        p_recall = sum(self.Y[self.Y>0]>self.threshold_positive)/sum(self.Y>0)
        n_recall = sum(self.Y[self.Y<0]<self.threshold_negative)/sum(self.Y<0)
        print("Positive threshold: {:.3f}, precision: {:.3f}, recall: {:.3f}\nNegative threshold: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(self.threshold_positive, self.positive_precision, p_recall, self.threshold_negative, self.negative_precision, n_recall))
        
    def set_positive_precision(self, k, delta = 0.01):
        p = 1.0
        start, end = 0.0, 1.0
        for i in range(100):
            l = (start + end) / 2
            p = sum(self.Y[self.Y_pred>l] > 0)/len(self.Y[self.Y_pred>l])
            if abs(p-k) < delta:
                self.threshold_positive = l
                self.positive_precision = p
                return
            elif p < k:
                start = l
            else:
                end = l
        self.threshold_positive = l
        self.positive_precision = p
        
    def set_negative_precision(self, k, delta = 0.01):
        p = 1.0
        start, end = -1.0, 0.0
        for i in range(100):
            l = (start + end) / 2
            p = sum(self.Y[self.Y_pred<l] < 0)/len(self.Y[self.Y_pred<l])
            if abs(p-k) < delta:
                self.threshold_negative = l
                self.negative_precision = p
                return
            elif p < k:
                end = l
            else:
                start = l
        self.threshold_negative = l
        self.negative_precision = p