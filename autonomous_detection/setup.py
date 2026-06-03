from setuptools import setup, find_packages

setup(
    name='autonomous_detection',
    version='0.1.0',
    description='SOTA autonomous object and vehicle detection pipeline',
    packages=find_packages(exclude=['notebooks', 'scripts', 'external']),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'ultralytics>=8.3.0',
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'Pillow>=10.0.0',
        'PyYAML>=6.0',
        'tqdm>=4.66.0',
    ],
    extras_require={
        'dev': ['pytest', 'black', 'isort'],
        'nuscenes': ['nuscenes-devkit>=1.1.10'],
        'augment': ['albumentations>=1.3.0'],
        'viz': ['matplotlib>=3.8.0', 'seaborn>=0.13.0'],
        'export': ['onnx>=1.14.0', 'onnxruntime>=1.16.0', 'onnxsim>=0.4.33'],
    },
)
