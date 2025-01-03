from typing import Dict


class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy) -> Dict:
        raise NotImplementedError()
