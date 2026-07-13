# SBEED datasets

This folder contains the replay buffers used by the SBEED solvers. They are
small data containers around PyTorch tensors, but they encode the transition
shape expected by the update code.

## Files

- `discrete_sbeed_dataset.py`: stores finite-MDP transitions as
  `(state, action, reward, next_state, done)`. It supports appending one
  transition, appending batches, FIFO truncation through a capacity, validation
  against the number of states/actions, and summaries used by experiment logs.
- `continuous_sbeed_dataset.py`: mirrors the discrete buffer for continuous
  observations and actions. Observations and actions are stored as float
  matrices, while rewards and terminal flags remain one-dimensional tensors.

Both buffers are used for online collection and for offline experiment data.
The one-step solvers sample individual rows. The multi-step solvers sample
contiguous fragments, so transition order and terminal flags are important:
fragments stop at `done=True` and do not cross episode boundaries.

