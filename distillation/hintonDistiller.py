import torch
import torch.nn as nn
from distillation.utils import Accuracy, AverageMeter, Hook
from distillation.baseDistiller import BaseDistiller

class HintonDistiller(BaseDistiller):
    def __init__(self, alpha, studentLayer=-2, teacherLayer=-2, n_samples_per_epoch = 390):
        super(HintonDistiller, self).__init__()
        
        self.alpha = alpha
        self.studentLayer = studentLayer
        self.teacherLayer = teacherLayer
        self.n_samples_per_epoch = n_samples_per_epoch
        
        # Register hooks
        self.studentHook = Hook()
        self.teacherHook = Hook()

    def train_step(self, student, teacher, dataloader, optimizer, objective, distillObjective, OneHot=False):
        """
        Train student model to the teacher model for one epoch with Hinton KD.
        
        :return: dict, named metrics for logging.
        """
        student.train()
        teacher.eval()
        
        # Attach
        if not self.studentHook.hooked():
            self._setHook(self.studentHook, student, self.studentLayer)
        if not self.teacherHook.hooked():
            self._setHook(self.teacherHook, teacher, self.teacherLayer)

        device = next(student.parameters()).device
        accuracy = Accuracy(OH=OneHot)
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        
        for num_iter, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Calculate logits
            sLogits = student(data)
            tLogits = teacher(data)
            
            # Retrieve activations from distillation layer of both models
            sAct = self.studentHook.val()
            tAct = self.teacherHook.val()
            
            # Calculate loss
            optimizer.zero_grad()
            batchLoss = (1-self.alpha)*distillObjective(nn.functional.log_softmax(sAct, dim=1), nn.functional.softmax(tAct, dim=1))
            batchLoss += self.alpha*objective(nn.functional.log_softmax(sLogits, dim=1), target)

            # Update student weights
            batchLoss.backward()
            optimizer.step()
            
            # Save metrics
            lossMeter.update(batchLoss.item(), n=len(data))
            accMeter.update(accuracy(nn.functional.softmax(sLogits, dim=1), target), n=len(data))

            #If the number of iterations = n_samples per epoch, break (parsed through the desired dataloader length)
            if num_iter == n_samples_per_epoch - 1: break
        
        return {'Train/Loss': lossMeter.avg,
                'Train/Accuracy': accMeter.avg}
