from setuptools import setup

setup(
    name='OTW',
    version='2.4',
    packages=['OTW', 'OTW.envs'],
    install_requires=['torch', 'gym', 'pandas', 'matplotlib', 'pygame', 'tensorboard', 'scipy', 'control'],
    url='',
    license='RML',
    author='Phat C. Vo',
    author_email='vophat0607@unist.ac.kr',
    description='On The Way (OTW) Env. for Optimal motion planing'
)
