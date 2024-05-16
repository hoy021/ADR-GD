import copy
import torch
import itertools
import numpy as np 
from torch.optim import Adam, Adagrad, RMSprop

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np 
import copy
# import torch.nn as nn

from models import *

def generate_params(shape,
               random_range = 1.,
               positive_prob = 0.5, 
               batch_norm = False):
    probs = np.random.uniform(0, 1, shape)
    W = np.random.uniform(0, random_range, shape)
    W[np.where(probs > positive_prob)] *= -1.
    if batch_norm:
        W /= np.linalg.norm(W, axis=0)
    return W

def generate_random_ReLU_NN(shapes,
                            random_range = 1.,
                            positive_prob = 0.5, 
                            batch_norm = False,
                            seed = None):
    np.random.seed(seed)
    
    Ws, bs = [], []
    for shape in shapes:
        W = generate_params(shape, random_range, 
                            positive_prob, batch_norm)
        b = generate_params(shape[1:], random_range,
                            positive_prob, batch_norm)
        Ws.append(W)
        bs.append(b)
    return Ws, bs

def estimate_ReLU_NN(x, Ws, bs):
    h = x.copy()
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = np.maximum(h @ W + b, 0.)
    return h @ Ws[-1] + bs[-1]

def input_optimization_gd(Ws_, bs_,
                          T, lr,
                          input_range = [-1., 1.],
                          return_history = False,
                          norm_grad = False):
                                
    Ws = [torch.Tensor(W).float().cuda() for W in Ws_]
    bs = [torch.Tensor(b).float().cuda() for b in bs_]
    x = torch.autograd.Variable(torch.zeros(Ws[0].shape[0], device = "cuda"),
                                requires_grad = True).float()
    x_history = []
    value_history = []
    grad_history = []

    relu = torch.nn.ReLU()
    best_value, best_x = - float('inf'), None
    for counter in range(T):

        h = x
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = h @ W + b 
            h = relu(h)
        y = - (h @ Ws[-1] + bs[-1])
        
        if x.grad is not None:
            x.grad.data.zero_()
        y.backward()

        x_grad = x.grad.data.clone()
        if norm_grad:
            x_grad /= x_grad.norm()
        x.data -= lr * (x_grad)
        x.data = torch.clamp(x, input_range[0], input_range[1])
        x_history.append(x.cpu().detach().numpy().copy())
        value_history.append((-y).cpu().detach().numpy().copy())
        grad_history.append(x.grad.cpu().detach().numpy().copy())
        
    if return_history:
        x_history = np.array(x_history)
        value_history = np.array(value_history)
        return -y.cpu().detach().numpy(), x.cpu().detach().numpy(), x_history, value_history, grad_history
    return -y.cpu().detach().numpy(), x.cpu().detach().numpy()

def input_optimization_ncopt(Ws_, bs_,
                             T, lr, lr_sigma, c, 
                             perturb_freq, noise_scale, grad_threshold,
                             lr_c = None,
                             input_range = [-1., 1.],
                             alpha = 2000., c_min = 0.2, c_max = 10.,
                             c_reduce_rate = 1e-2, c_grad_tolerance = 1e-2,
                             x_grad_threshold = 1., 
                             set_proper_sigma_freq = None,
                             apply_normalization = True):
    Ws = [torch.Tensor(W).float().cuda() for W in Ws_]
    bs = [torch.Tensor(b).float().cuda() for b in bs_]
        
    x = torch.autograd.Variable(torch.zeros(Ws_[0].shape[0], device = "cuda"),
                                requires_grad = True).float()
    c = torch.autograd.Variable(torch.tensor(c, device = "cuda"),
                                requires_grad = True).float()
    lam_sigmoid = lambda x: torch.sigmoid(alpha * x)
    
    def set_diag_simga():
        diag_sigma_list = []
        
        h = x 
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = h @ W + b
            sign_h = torch.sign(h).detach()
            diag_sigma = torch.autograd.Variable((0.5 - sign_h * 1e-2), requires_grad=True).float()
            diag_sigma_list.append(diag_sigma)
            h = torch.maximum(h, torch.zeros(h.shape).cuda())
        return diag_sigma_list
     
    diag_sigma_list = set_diag_simga()
    
    x_history = []
    last_perturb_iter = [0 for _ in range(len(Ws) - 1)]
    for counter in range(T):
        if set_proper_sigma_freq is not None and (counter + 1) % set_proper_sigma_freq == 0:
            diag_sigma_list = set_diag_simga()
        constraint_loss = 0.
        x_constraint_loss = 0.
        h_relu = x
        h = x
        prod_W = None
        for W, b, diag_sigma in zip(Ws[:-1], bs[:-1], diag_sigma_list):
            diag_sigma_activations = 1 - lam_sigmoid(diag_sigma - 0.5 + 1e-6)
            assert not torch.any(torch.isnan(diag_sigma_activations))

            I_sigma = torch.diag(diag_sigma_activations)
            targets = h_relu @ W + b
            if prod_W is None:
                prod_W = W
            else:
                prod_W = prod_W @ W
            if apply_normalization:
                targets = targets / torch.linalg.norm(prod_W, axis = 0)
            prod_W = prod_W @ I_sigma.detach()
            
            prob = diag_sigma

            x_constraint_loss1 = torch.maximum(targets  * torch.maximum(prob - 0.5,
                                                    torch.zeros(diag_sigma.shape).cuda()),
                                             torch.zeros(diag_sigma.shape).cuda()).max()
            x_constraint_loss2 = torch.maximum((- targets ) * torch.maximum(0.5 - prob,
                                                    torch.zeros(diag_sigma.shape).cuda()),
                                             torch.zeros(diag_sigma.shape).cuda()).max()
            x_constraint_loss += x_constraint_loss1 + x_constraint_loss2
            
            h = (h @ W + b) @ I_sigma
            h_relu = h_relu @ W + b 
            h_relu = torch.maximum(h_relu, torch.zeros(h_relu.shape).cuda())
            
        if apply_normalization:
            y = - (h @ Ws[-1] + bs[-1]) / torch.linalg.norm(prod_W @ Ws[-1])
        else:
            y = - (h @ Ws[-1] + bs[-1])
        y += c * x_constraint_loss
      
        if x.grad is not None:       
            x.grad.data.zero_()
            if c.grad is not None:
                c.grad.data.zero_()
            c.grad.data.zero_()
            for counter_sigma in range(len(diag_sigma_list)):
                if diag_sigma_list[counter_sigma].grad is not None:
                    diag_sigma_list[counter_sigma].grad.data.zero_()
                    
        y.backward()
        x_grad = x.grad.data.clone()
        if (not torch.isclose(x_grad.norm(), torch.zeros(x_grad.norm().shape).cuda(),
                             atol = 1e-4)) and (x_grad_threshold is None or x_grad.norm() > x_grad_threshold):
            x_grad /= x_grad.norm()
                
        x.data -= lr * (x_grad)
        x.data = torch.clamp(x, input_range[0], input_range[1])
        x_history.append(x.cpu().detach().numpy().copy())

        if lr_c is not None and c.grad is not None:
            if c.grad.norm() <= c_grad_tolerance:
                c.data -= c_reduce_rate
            else:
                c.data += lr_c * c.grad
            c.data = torch.clamp(c, c_min, c_max)
                
        for counter_sigma in range(len(diag_sigma_list)):
            grad = diag_sigma_list[counter_sigma].grad.data.clone()
            if not torch.isclose(grad.norm(), torch.zeros(grad.norm().shape).cuda(),
                 atol = 1e-4) and grad.norm() > 1.:
                grad /= grad.norm()
            if perturb_freq is not None and (counter - last_perturb_iter[counter_sigma]) >= perturb_freq and grad.norm() <= grad_threshold:
                perturbations = torch.randn(grad.shape).cuda() * noise_scale
                grad += perturbations.cuda()
                last_perturb_iter[counter_sigma] = counter
            
            diag_sigma_list[counter_sigma].data -= lr_sigma * grad 
            diag_sigma_list[counter_sigma].data = torch.clamp(diag_sigma_list[counter_sigma], 0.3, 0.7)
                    
    x_history = np.array(x_history)
    values = estimate_ReLU_NN(x_history, Ws_, bs_)      
    return np.max(values), x_history[np.argmax(values)]

def load_MNIST_dataset(batch_size = 128,
                       shuffle = False):
    mnist_train = dsets.MNIST(root = './data/',
                              train = True,
                              transform = transforms.Compose(
                                        [transforms.Resize(28), transforms.ToTensor()]
                                    ),
                              download = True)

    mnist_test = dsets.MNIST(root = './data/',
                             train=False,
                             transform = transforms.Compose(
                                        [transforms.Resize(28), transforms.ToTensor()]
                                    ),
                             download = True)

    train_loader  = torch.utils.data.DataLoader(dataset = mnist_train,
                                               batch_size = batch_size,
                                               shuffle = shuffle)

    test_loader = torch.utils.data.DataLoader(dataset = mnist_test,
                                              batch_size = batch_size,
                                              shuffle = shuffle)
    
    return train_loader, test_loader, len(mnist_train)

def convert_convolutional_kernel_to_FC(img_shape, kernel,
                                       stride = 1, 
                                       padding = 0):
    assert img_shape[-1] == kernel.shape[-1]
    assert img_shape[0] == img_shape[1] # only allows square matrix atm 
    H, _, d = img_shape
    F, _, _ = kernel.shape
    
    out_dim = int((H + 2 * padding - F) / stride) + 1
    flatten_out_dim = out_dim * out_dim * 1# d
    assert H * H * d == np.prod(img_shape)
    flatten_dim = ((H + 2 * padding) ** 2) * d
    FC_weights = np.zeros([flatten_dim, flatten_out_dim])
    
    H_ = H + 2 * padding
    for d_counter in range(d):
        tmp_y = out_y = 0
        while tmp_y + F <= H_:
            tmp_x = out_x = 0
            while tmp_x + F <= H_:

                patch_indices = []
                for tmp_y_ in range(tmp_y, (tmp_y + F)):
                    for tmp_x_ in range(tmp_x, (tmp_x + F)):
                        patch_indices.append(tmp_x_ + tmp_y_ * H_ + d_counter * H_ * H_)
                patch_indices = np.array(patch_indices)

                out_idx = out_x + out_y * out_dim

                FC_weights[:, out_idx][patch_indices] += kernel[:, :, d_counter].reshape([-1]).copy()
                
                tmp_x += stride 
                out_x += 1 
            tmp_y += stride 
            out_y += 1 
            

    if padding != 0:
        padded_indices = []
        for d_counter in range(d):
            for tmp_y in range(padding):
                tmp_xs = np.arange(0, H_)
                padded_indices.append(tmp_xs + tmp_y * H_ + d_counter * H_ * H_) 
            for tmp_y in range(H_ - 1, H_ - padding - 1, -1):
                tmp_xs = np.arange(0, H_)
                padded_indices.append(tmp_xs + tmp_y * H_ + d_counter * H_ * H_) 
            for tmp_x in range(padding):
                tmp_ys = np.arange(0, H_)
                padded_indices.append(tmp_x + tmp_ys * H_ + d_counter * H_ * H_) 
            for tmp_x in range(H_ - 1, H_ - padding - 1, -1):
                tmp_ys = np.arange(0, H_)
                padded_indices.append(tmp_x + tmp_ys * H_ + d_counter * H_ * H_) 
        padded_indices = np.unique(np.concatenate(padded_indices))
        keep_indices = np.delete(np.arange(flatten_dim),
                                 padded_indices)
        FC_weights = FC_weights[keep_indices]
        
    return FC_weights

def convert_conv2d_to_FC(conv_layer, img_shape,
                         padding = 0):

    current_W, current_b = [], []

    kernels = conv_layer.weight.cpu().detach().numpy()
    kernel_biases = conv_layer.bias.cpu().detach().numpy()
    output_img_shape = None
        
    for kernel, kernel_bias in zip(kernels, kernel_biases):
        kernel_temp = np.swapaxes(kernel.copy(), 0, 1)
        kernel_temp = np.swapaxes(kernel_temp, 1, 2)
        W_conv = convert_convolutional_kernel_to_FC(img_shape,
                                                    kernel_temp,
                                                    padding = padding)
        b_conv = np.ones(W_conv.shape[1]) * kernel_bias
        current_W.append(W_conv)
        current_b.append(b_conv)
        
        if output_img_shape is None:
            output_img_shape = int(np.sqrt(W_conv.shape[1]))
                        
    current_W = np.hstack(current_W)
    current_b = np.concatenate(current_b)
    return current_W, current_b, output_img_shape, kernels.shape[0]

def convert_CNN_to_FC(model, img_shape,
                      fc_dims = [100, 10]):

    current_img_size = img_shape[0]
    current_img_dim = img_shape[-1]
    Ws, bs = [], []
    FC_dims = []

    for layer in model.ConvLayers:
        if isinstance(layer, nn.Conv2d):
            W, b, new_img_size, new_img_dim = convert_conv2d_to_FC(layer,
                                                      [current_img_size, current_img_size, current_img_dim])
            current_img_size = new_img_size
            current_img_dim = new_img_dim
            Ws.append(W)
            bs.append(b)
            
            if len(FC_dims) == 0:
                FC_dims += list(W.shape)
            else:
                FC_dims.append(W.shape[1])
#             print (W.shape, b.shape)

    FC_dims += fc_dims
    assert len(FC_dims) > 2
    fc_model = FCNN(FC_dims[0], FC_dims[-1],
                    FC_dims[1:-1])
    
    # transfering the model weights 
    with torch.no_grad():
        for counter, (W, b) in enumerate(zip(Ws, bs)):
            fc_model.layers[counter * 2].weight = nn.Parameter(torch.Tensor(W).float().cuda().T)
            fc_model.layers[counter * 2].bias = nn.Parameter(torch.Tensor(b).float().cuda())
            
        for sub_counter in range(len(fc_dims)):
            counter += 1 
            fc_model.layers[counter * 2].weight = nn.Parameter(model.fc_layer[sub_counter * 2].weight.clone().cuda())
            fc_model.layers[counter * 2].bias = nn.Parameter(model.fc_layer[sub_counter * 2].bias.clone().cuda())
        
    return fc_model 

def extract_Ws_and_bs_from_fc_model(fc_model):
    Ws_tensor, bs_tensor = [], []
    Ws, bs = [], []
    for layer in fc_model.layers:
        if isinstance(layer, nn.Linear):
            Ws_tensor.append(layer.weight.clone().float().cuda())
            bs_tensor.append(layer.bias.clone().float().cuda())
            Ws.append(layer.weight.clone().float().cpu().detach().numpy())
            bs.append(layer.bias.clone().float().cpu().detach().numpy())
    return Ws_tensor, bs_tensor, Ws, bs

def estimate_ReLU_NN_MNIST(x, Ws, bs, exclude_ReLU = False,
                     last_layer_abs = False,
                     abs_weights = False,
                     transpose_W = False,
                     disconnect_relu_layer_ids = []):
    Ws_temp = copy.deepcopy(Ws)
    if transpose_W:
        Ws_temp = [W.T for W in Ws_temp]
        
    h = x.copy()
    if abs_weights:
        # print ("hi")
        h = h.dot(np.abs(Ws_temp[0])) + bs[0]
    else:
        h = h.dot(Ws_temp[0]) + bs[0]
    for counter, (W, b) in enumerate(zip(Ws_temp[1:-1], bs[1:-1])):
        if abs_weights:
            h = np.maximum(h, 0).dot(np.abs(W)) + b
        elif counter not in disconnect_relu_layer_ids:
            h = np.maximum(h, 0).dot(W) + b
        else:
            h = h.dot(W) + b

    if last_layer_abs:
        h = np.abs(h)
    else:
        h = np.maximum(h, 0)
    if abs_weights:
        return h.dot(np.abs(Ws_temp[-1])) + bs[-1]
    else:
        return h.dot(Ws_temp[-1]) + bs[-1]


def adversarial_attack_nonconvexOpt(image, label, fc_model, epsilon,
                                    T, lr, lr_sigma,
                                    lr_c = None,
                                    c = 1.,
                                    perturb_freq = None,
                                    verbose_freq = None,
                                    set_proper_sigma_freq = 16,
                                    noise_scale = 0.1,
                                    grad_threshold = 1e-1,
                                    c_reduce_rate = 1e-2, c_grad_tolerance = 1e-2, c_min = 0.2, c_max = 10.,
                                    target_label = None):


    entropy_loss = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=0)
    Ws_tensor, bs_tensor, Ws, bs = extract_Ws_and_bs_from_fc_model(fc_model)
    _, d0 = Ws_tensor[0].shape
    x = torch.autograd.Variable(torch.zeros(d0, device = "cuda"), requires_grad=True,).float()
    c = torch.autograd.Variable(torch.tensor(c, device = "cuda"),
                                requires_grad = True).float()
    
    alpha = 2000.
    lam_sigmoid = lambda x: torch.sigmoid(alpha * x)
    x_low, x_high = torch.zeros(x.shape).cuda(), torch.ones(x.shape).cuda().float()

    def set_diag_sigma():
        diag_sigma_list = []

#         print (x.shape, image.shape)
        h = x + image
                        
        for W, b in zip(Ws_tensor[:-1], bs_tensor[:-1]):
            h = W @ h + b
            sign_h = torch.sign(h).detach()
            d = W.shape[0]
            diag_sigma = torch.autograd.Variable((0.5 - sign_h * 1e-2).cuda(), requires_grad=True).float()
            diag_sigma_list.append(diag_sigma)

            h = torch.maximum(h, torch.zeros(h.shape).cuda())
        return diag_sigma_list

    diag_sigma_list = set_diag_sigma()
    best_value, best_x = -float('inf'), None
    last_perturb_iter = [0 for _ in range(len(diag_sigma_list))]
    for counter in range(T):
        
        if set_proper_sigma_freq is not None and (counter + 1) % set_proper_sigma_freq == 0:
            diag_sigma_list = set_diag_sigma()

        h = (x + image)
        h_relu = (x + image)
        constraint_loss = 0
        prod_W = None
        for counter_plane, (W_tensor, b_tensor, diag_sigma) in enumerate(zip(Ws_tensor[:-1], bs_tensor[:-1], diag_sigma_list)):
            if prod_W is None:
                prod_W = W_tensor
            else:
                prod_W = W_tensor @ prod_W
                
            diag_sigma_activations = 1 - lam_sigmoid(diag_sigma - 0.5 + 1e-6)#.detach()
#             assert not torch.any(torch.isnan(diag_sigma_activations))

            I_sigma = torch.diag(diag_sigma_activations)
            prod_W = I_sigma.detach() @ prod_W 
            
            targets = W_tensor @ h_relu + b_tensor            
            prod_W_norm = torch.linalg.norm(prod_W, axis = 1)
            targets = targets / torch.maximum(prod_W_norm, 
        torch.ones(prod_W_norm.shape).cuda())
            
            prob = diag_sigma
            constraint_loss1 = torch.maximum(targets  * torch.maximum(prob - 0.5,
                                                    torch.zeros(diag_sigma.shape).cuda()),
                                             torch.zeros(diag_sigma.shape).cuda()).max()
            constraint_loss2 = torch.maximum((- targets ) * torch.maximum(0.5 - prob,
                                                    torch.zeros(diag_sigma.shape).cuda()),
                                             torch.zeros(diag_sigma.shape).cuda()).max()
            constraint_loss += constraint_loss1 + constraint_loss2

            h = I_sigma @ (W_tensor @ h + b_tensor)
                
            h_relu = W_tensor @ h_relu + b_tensor
            h_relu = torch.maximum(h_relu, torch.zeros(h_relu.shape).cuda())
        y = Ws_tensor[-1] @ h + bs_tensor[-1]
        y /= y.norm().detach()
        if target_label is None:
            loss = - entropy_loss(y, label)
        else:
            loss = entropy_loss(y, target_label)
        loss += c * constraint_loss
            
#         assert torch.all(torch.isfinite(constraint_loss))

        if x.grad is not None:
            x.grad.data.zero_()
            for counter_sigma in range(len(diag_sigma_list)):
                if diag_sigma_list[counter_sigma].grad is not None:
                    diag_sigma_list[counter_sigma].grad.data.zero_()

        loss.backward()

        x_grad = x.grad.data.clone()
        if (not torch.isclose(x_grad.norm(), torch.zeros(x_grad.norm().shape).cuda(),
                     atol = 1e-4)) and x_grad.norm() > 1.:
            x_grad /= x_grad.norm()
        x.data -= lr * (x_grad)
        x.data = torch.clamp(x, - epsilon, epsilon)
        
        if lr_c is not None and c.grad is not None:
            if c.grad.norm() <= c_grad_tolerance:
                c.data -= c_reduce_rate
            else:
                c.data += lr_c * c.grad
            c.data = torch.clamp(c, c_min, c_max)
            

        diag_sigma_grads = []
        for counter_sigma in range(len(diag_sigma_list)):

            grad = diag_sigma_list[counter_sigma].grad.data.clone()
            if not torch.isclose(grad.norm(), torch.zeros(grad.norm().shape).cuda(),
                     atol = 1e-4) and grad.norm() > 1.:
                grad /= grad.norm()
            if perturb_freq is not None and (counter - last_perturb_iter[counter_sigma]) >= perturb_freq and grad.norm() <= grad_threshold:
                perturbations = torch.randn(grad.shape) * noise_scale 
                grad += perturbations.cuda()
                last_perturb_iter[counter_sigma] = counter

                test_val = entropy_loss(torch.Tensor(estimate_ReLU_NN_MNIST((image + x).cpu().detach().numpy(), Ws, bs, transpose_W = True)).cuda(), 
                                         label)
                if test_val > best_value:        
                    best_x, best_value = x.clone(), test_val
                    
            diag_sigma_list[counter_sigma].data -= lr_sigma * grad 
            diag_sigma_list[counter_sigma].data = torch.clamp(diag_sigma_list[counter_sigma], 0.3, 0.7)

        if verbose_freq is not None and (counter + 1) % verbose_freq == 0:
            print ("Iter: {}, actual y: {} \n\tconstraint_loss: {}\n".format(counter + 1, best_value, constraint_loss))

    del x 
    for diag_sigma in diag_sigma_list:
        del diag_sigma
    return image + best_x, best_value

def draw_image_randomly(test_loader, prob = 0.3):
    for images, labels in test_loader:
        if np.random.uniform(0, 1) <= prob: 
            break
    idx = np.random.randint(images.shape[0])
    return images[idx].cuda(), labels[idx].cuda()