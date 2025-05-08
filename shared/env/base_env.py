from abc import ABC, abstractmethod

class BaseLowdimRunner(ABC):
    """
    Base class for low-dimensional environment runners.
    This abstract class defines the interface that all environment runners should implement.
    """
    
    def __init__(self):
        """Initialize the environment runner"""
        pass
        
    @abstractmethod
    def run(self, policy):
        """
        Run the policy on the environment
        
        Args:
            policy: Policy to run
            
        Returns:
            Dictionary with run statistics
        """
        pass 