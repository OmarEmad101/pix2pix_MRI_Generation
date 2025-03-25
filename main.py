import subprocess
import os

# Base directory (where the script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"D:\Studying\Fourth year\GP\Datasets\Final Data\All Data Resampled\EMIDEC_original_size\final_localization_of_the_full_augmented_data\model_data"


models = [
    # {"script":os.path.join("models", "basic_model_lambda_80.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_lambda_90.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_lambda_100.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_lambda_110.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_lambda_120.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_weighted_loss_lambda_80.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_weighted_loss_lambda_90.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_weighted_loss_lambda_100.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_weighted_loss_lambda_110.py"),"apply_noise": False},
    # {"script":os.path.join("models", "basic_model_weighted_loss_lambda_120.py"),"apply_noise": False},
    {"script":os.path.join("models", "noisy_model_weighted_loss_lambda_80.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.4,},
    # {"script":os.path.join("models", "noisy_model_weighted_loss_lambda_90.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.3},
    # {"script":os.path.join("models", "noisy_model_weighted_loss_lambda_100.py"),"apply_noise": True, "noise_std": 0.1, "noise_max": 0.3},
    # {"script":os.path.join("models", "noisy_model_weighted_loss_lambda_110.py"),"apply_noise": True, "noise_std": 0.1, "noise_max": 0.2},
    # {"script":os.path.join("models", "noisy_model_weighted_loss_lambda_120.py"),"apply_noise": True, "noise_std": 0.1, "noise_max": 0.2},
    # {"script":os.path.join("models", "noisy_modified_disc_weighted_loss_lambda_70.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.3, "conditional_disc": True},
    # {"script":os.path.join("models", "noisy_modified_disc_weighted_loss_lambda_80.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.3, "conditional_disc": True},
    # {"script":os.path.join("models", "noisy_modified_disc_weighted_loss_lambda_90.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.3, "conditional_disc": True},
    {"script":os.path.join("models", "noisy_modified_disc_weighted_loss_lambda_100.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.3, "conditional_disc": True},
    # {"script":os.path.join("models", "noisy_modified_disc_weighted_loss_lambda_110.py"),"apply_noise": True, "noise_std": 0.2, "noise_max": 0.3, "conditional_disc": True},

]

for model in models:
    env_vars = {
        **os.environ,
        "DATA_DIR": DATA_DIR,
        "APPLY_NOISE": str(model["apply_noise"]),
        "NOISE_STD": str(model.get("noise_std", 0.2)),
        "NOISE_MAX": str(model.get("noise_max", 0.5)),
        "CONDITIONAL_DISC": str(model.get("conditional_disc", False))
    }
    print(f"Starting training for {model['script']} (Noise: {model['apply_noise']})...")
    subprocess.run(["python", model["script"]], env=env_vars)
    print(f"Finished training for {model['script']}.\n")



