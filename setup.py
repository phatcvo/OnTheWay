from setuptools import setup

setup(
    name='OTW',
    version='3.1',
    packages=['OTW', 'OTW.envs'],
    install_requires=['torch', 'gym', 'pandas', 'matplotlib', 'pygame', 'tensorboard', 'scipy', 'control', 'stable-baselines3'],
    python_requires=">=3.8",
    license='MIT',
    description='Simulation Environment for RL-based Autonomous Driving'
)
