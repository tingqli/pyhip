from setuptools import setup, find_packages

setup(
    name='pyhip',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A python interface for ROCM HIP language',
    url='https://github.com/tingqli/pyhip',
    author='Li, Tingqian',
    author_email='ltq18@hotmail.com',
    install_requires=['filelock','numpy']
)