## Activation-Descent Regularization Gradient Descent (ADR-GD)

Proof of concept for ICML'24 "Activation-Descent Regularization for Input Optimization of ReLU Networks"

Installation
--- 
With [Ananoconda3](https://docs.anaconda.com/free/anaconda/install/index.html) installed, run the command `bash install.sh`.

Run `conda activate ADRGD` to activate the environment with required packages configured.

Quickstart
---

- `randomized_network_input_optimization.ipynb` - Input optimization to maximize the outputs of randomized ReLU networks.

- `adversarial_optimization.ipynb` - Adversarial optimization for MNIST over clean and adversarially-trained image classifiers.

- See the folder `enhanced_RL` for scripts to enhance the training of actor-critic-based DRL via formulating input optimization problems to improve exploitation actions.

Future Todos
--- 

- Will upload the clean codes of adversarial optimization for CIFAR10 & ImageNet in the recent weeks.

- Extend/Explore ADR-GD to the networks with the other activation functions.
  - Motivation driven by the piecewise-continuous value-landscape from these networks, in comparison to the piecewise-linear landscape induced by ReLU networks.
