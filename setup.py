from setuptools import setup, find_packages

setup(
    name='masters_thesis',
    version='0.1.0',
    author='Yuto Kvist',
    author_email='yuto.kvist@abo.fi',
    description='Masters thesis',
    long_description=open('README.md').read(),
    url='https://github.com/yourusername/yourproject',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'mpi4py',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)