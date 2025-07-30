import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        """
        student_logits: output from the student model
        teacher_logits: output from the teacher model
        targets: ground truth labels
        """
        # Hard target loss
        loss_ce = self.ce_loss(student_logits, targets)

        # Soft target loss
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        loss_kl = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

        # Combined loss
        loss = self.alpha * loss_kl + (1. - self.alpha) * loss_ce
        return loss


# Utility function for distillation training
def distillation_step(student_model, teacher_model, data, targets, optimizer, kd_criterion):
    student_model.train()
    teacher_model.eval()

    with torch.no_grad():
        teacher_output = teacher_model(data)

    student_output = student_model(data)
    loss = kd_criterion(student_output, teacher_output, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
