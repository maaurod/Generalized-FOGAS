# SBEED features and parametrizations

This folder defines the value, rho, and policy parametrizations used by the
SBEED implementation.

## Discrete features

- `TabularStateFeatures` and `TabularStateActionFeatures` create one-hot
  features for small finite MDPs. They remove representation error and are used
  for sanity checks on gridworlds.
- `RBFStateFeatures` creates radial-basis features from state coordinates,
  typically normalized grid coordinates. `RBFStateActionFeatures` expands those
  features into one action block per action for the rho model.
- `LinearValueParam`, `LinearRhoParam`, and `SoftmaxLinearPolicyParam` wrap
  feature maps as PyTorch modules so the final solver can use a common update
  interface.
- `NeuralValueParam`, `NeuralRhoParam`, and `NeuralPolicyParam` wrap neural
  modules for nonlinear discrete experiments.

## Continuous parametrizations

- `ContinuousStateMLPValueModule` and `ContinuousStateActionMLPModule` build the
  neural networks used for continuous value and rho models.
- `ContinuousNeuralValueParam` and `ContinuousNeuralRhoParam` adapt those
  networks to the solver interface.
- `RFFGaussianPolicyParam` implements the Gaussian policy used in continuous
  experiments. Its mean is represented with random Fourier features, and the
  policy exposes sampling, log-probability, and KL operations used by the
  natural-gradient policy update.

