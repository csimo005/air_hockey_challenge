from stable_baselines3.common.callbacks import BaseCallback
import os

vecnormalize_file_name = "vecnormalize.pkl"

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        path = os.path.join(self.save_path, "vecnormalize.pkl")

        self.model.get_vec_normalize_env().save(path)
        if self.verbose > 1:
            print(f"Saving VecNormalize to {path}")

        return True