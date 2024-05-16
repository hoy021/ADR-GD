model_path = "models/mnist_pgd_epsilon_0.25_steps_20_epochs_20"

epsilon = 0.25
T = 500
lr = 10e-1 
lr_c = 1e-4
lr_sigma = 2.5e-1 
c = 0.5
perturb_freq = 5
noise_scale = 1e-3
grad_threshold = 1e-5
c_reduce_rate = 1e-3 

hiddens_config = [
    [1, 8, 10],
    [8, 32, 8],
    [32, 32, 8],  
]
batch_size = 64