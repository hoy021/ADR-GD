import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, input_dim,
                 hiddens_config,
                 fc_hiddens_config = [100, 10]):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.hiddens_config = hiddens_config
        self.conv_latent_dim = input_dim
        self.conv_latent_size = input_size
        
        self.ConvLayers = []
        for hidden_params in self.hiddens_config:
            assert len(hidden_params) == 3
            in_dim, out_dim, kernel_dim = hidden_params
            self.ConvLayers.append(nn.Conv2d(in_dim, out_dim, kernel_dim).cuda())
            self.ConvLayers.append(nn.ReLU().cuda())
            self.conv_latent_dim = out_dim
            self.conv_latent_size -= (kernel_dim - 1)
            assert self.conv_latent_size > 0
#             print (self.conv_latent_dim * (self.conv_latent_size ** 2))
        self.ConvLayers = nn.Sequential(*self.ConvLayers)
            
        self.conv_final_dim = self.conv_latent_dim * (self.conv_latent_size ** 2)
        self.fc_layer = []
        current_dim = self.conv_final_dim
        for h in fc_hiddens_config:
            self.fc_layer.append(nn.Linear(current_dim, h).cuda())
            self.fc_layer.append(nn.ReLU().cuda())
            current_dim = h
        self.fc_layer = nn.Sequential(*(self.fc_layer[:-1]))       
        
    def forward(self,x):
        out = self.ConvLayers(x)
        out = out.view(-1, self.conv_final_dim)
        out = self.fc_layer(out)

        return out
    
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens_config,
                 disconnect_relu_layer_ids = []):
        super(FCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.hiddens_config = hiddens_config 
        self.disconnect_relu_layer_ids = disconnect_relu_layer_ids
        
        layers = []
        current_dim = self.input_dim
        for counter, h in enumerate(hiddens_config):
            layers.append(nn.Linear(current_dim, h,).cuda())
            if counter not in self.disconnect_relu_layer_ids:
                layers.append(nn.ReLU().cuda())
            current_dim = h 
        layers.append(nn.Linear(current_dim, self.output_dim,).cuda())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    

    