# Fast Gradient Sign Method
- first introduced by Goodfellow 2014
- to perturb the input data by adding a small amount of noise in the direction of the gradient of the loss function with respect to the input

Adversarial example: x_adv = x + ε * sign(∇x J(θ, x, y))

where:
- x_adv is the adversarial example
- x is the original input
- ε is a small perturbation factor that controls the magnitude of the perturbation
- ∇x is the gradient of the loss function J with respect to the input x
- J(θ, x, y) is the loss function, which depends on the model parameters θ, the input x, and the true label y

The sign function in the formula ensures that the perturbation is added in the direction that maximizes the loss function. By adding this perturbation to the original input, the resulting adversarial example can cause the model to misclassify the input.

1. Compute the gradient of the loss function with respect to the input: ∇x J(θ, x, y)
2. Calculate the perturbation by multiplying the gradient by the perturbation factor: ε * sign(∇x J(θ, x, y))
3. Add the perturbation to the original input: x_adv = x + ε * sign(∇x J(θ, x, y))
4. Use the adversarial example x_adv to evaluate the model's performance and analyze its vulnerability to adversarial attacks.
