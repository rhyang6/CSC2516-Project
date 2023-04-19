# adapted from torchattacks
# Hoki Kim. Torchattacks : A pytorch repository for adversarial attacks. CoRR, abs/2010.01950,2020.

import torch

from utils.loss import ComputeLoss

class Attack:
    """
    Base class for all attacks
    """
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, images, labels=None, *args, **kwargs):
        adv_images = self.forward(images, labels, *args, **kwargs)
        return adv_images

class PGD(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def forward(self, images, targets):
        adv_images = images.detach()
        targets = targets.detach()

        compute_loss = ComputeLoss(self.model)

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            out, train_out = self.model(adv_images)

            # Calculate loss
            loss, loss_items = compute_loss(train_out, targets)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class FGSM(Attack):
    def __init__(self, model, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps

    def forward(self, images, targets):
        images = images.detach()
        targets = targets.detach()
        
        compute_loss = ComputeLoss(self.model)

        images.requires_grad = True
        out, train_out = self.model(images)

        # Calculate loss
        loss, loss_items = compute_loss(train_out, targets)

        # Update adversarial images
        grad = torch.autograd.grad(loss, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

class MIFGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha

    def forward(self, images, targets):
        images = images.detach()
        targets = targets.detach()
        
        momentum = torch.zeros_like(images).detach()

        compute_loss = ComputeLoss(self.model)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            out, train_out = self.model(adv_images)

            # Calculate loss
            loss, loss_items = compute_loss(train_out, targets)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad),
                                     dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

# class DeepFool(Attack):
#     def __init__(self, model, steps=50, overshoot=0.02):
#         super().__init__("DeepFool", model)
#         self.steps = steps
#         self.overshoot = overshoot

#     def forward(self, images, targets):
#         adv_images, target_labels = self.forward_return_target_labels(images, targets)
#         return adv_images


#     def forward_return_target_labels(self, images, targets):
#         images = images.detach()
#         targets = targets.detach() # nbboxes, (batch_index, cls, x, y, w, h)
        
#         batch_size = len(images)
#         correct = torch.tensor([True]*batch_size)
#         lb = [targets[targets[:, 0] == i, 1:] for i in range(batch_size)] # [batch_index, tensor(nbboxes_per_batch_index, (cls, x, y, w, h))]
#         target_labels = [targets[targets[:, 0] == i, 1:] for i in range(batch_size)] # [batch_index, tensor(nbboxes_per_batch_index, (cls, x, y, w, h))]
#         targets = lb
#         # target_labels = targets.clone().detach()
#         curr_steps = 0

#         adv_images = []
#         for idx in range(batch_size):
#             image = images[idx:idx+1].clone().detach()
#             adv_images.append(image)

#         while (True in correct) and (curr_steps < self.steps):
#             for idx in range(batch_size):
#                 if not correct[idx]: continue
#                 early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], targets[idx])
#                 adv_images[idx] = adv_image
#                 target_labels[idx] = pre
#                 if early_stop:
#                     correct[idx] = False
#             curr_steps += 1

#         adv_images = torch.cat(adv_images).detach()
#         return adv_images, target_labels


#     def _forward_indiv(self, image, label):
#         image.requires_grad = True
#         out, train_out = self.model(image)
#         print(out.shape, [o.shape for o in train_out])
#         fs = self.model(image)[0][0]
#         # print(fs.shape)
#         _, pre = torch.max(fs, dim=0)
#         print(pre.shape)
#         if pre != label:
#             return (True, pre, image)

#         ws = self._construct_jacobian(fs, image)
#         image = image.detach()

#         f_0 = fs[label]
#         w_0 = ws[label]

#         wrong_classes = [i for i in range(len(fs)) if i != label]
#         f_k = fs[wrong_classes]
#         w_k = ws[wrong_classes]

#         f_prime = f_k - f_0
#         w_prime = w_k - w_0
#         value = torch.abs(f_prime) \
#                 / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
#         _, hat_L = torch.min(value, 0)

#         delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
#                  / (torch.norm(w_prime[hat_L], p=2)**2))

#         target_label = hat_L if hat_L < label else hat_L+1

#         adv_image = image + (1+self.overshoot)*delta
#         adv_image = torch.clamp(adv_image, min=0, max=1).detach()
#         return (False, target_label, adv_image)

#     # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
#     # torch.autograd.functional.jacobian is only for torch >= 1.5.1
#     def _construct_jacobian(self, y, x):
#         x_grads = []
#         for idx, y_element in enumerate(y):
#             if x.grad is not None:
#                 x.grad.zero_()
#             y_element.backward(retain_graph=(False or idx+1 < len(y)))
#             x_grads.append(x.grad.clone().detach())
#         return torch.stack(x_grads).reshape(*y.shape, *x.shape)
