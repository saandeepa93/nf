from yacs.config import CfgNode as CN

_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.DATA_ROOT = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.VIS_PATH = ""

# FLOW
_C.FLOW = CN()
_C.FLOW.N_FLOW = 8
_C.FLOW.N_BLOCK = 1
_C.FLOW.N_CHAN = 2
_C.FLOW.MLP_DIM = 23

# DATASET
_C.DATASET = CN()
_C.DATASET.N_CLASS = 2

# LOSS
_C.LOSS = CN()
_C.LOSS.LAMBDA = 0.0051

# TRAINING
_C.TRAINING = CN()
_C.TRAINING.ITER = 1000
_C.TRAINING.BATCH = 256
_C.TRAINING.LR = 1e-3
_C.TRAINING.WT_DECAY = 1e-5


# COMMENTS
_C.COMMENTS = "TEST"




def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()