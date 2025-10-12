from setuptools import setup, find_packages

setup(
    name="otw_env",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "gymnasium",
        "matplotlib",
        "pygame",
        "pyyaml",
        "tqdm",
        "stable-baselines3>=2.3.0",
        "torch>=2.0",
        "tensorboard",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "street-v1 = otw_env.envs:StreetEnv"
        ],
    },
)
