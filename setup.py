from setuptools import find_packages, setup

setup(
    name = 'forecast-team-1',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'numpy==1.19.4',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'sklearn==0.0',
        'matplotlib==3.3.3',
        'PythonDataProcessing @ git+https://github.com/Deep-Stonks-Group/PythonDataProcessing.git',
    ],
)
