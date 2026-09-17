"""BrainMemory: biologically inspired experimental memory via synaptic change."""

from brainmemory.autonomy.learner import AutonomousLearner
from brainmemory.brain import Brain
from brainmemory.config import BrainConfig

__version__ = "0.1.0"
__all__ = ["Brain", "BrainConfig", "AutonomousLearner", "__version__"]
