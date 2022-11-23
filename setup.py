from setuptools import setup, find_packages
from glob import glob

setup(
    name='DashAI',
    packages=find_packages(),
    include_package_data=True,
    # data_files=[('DashAI/front/build', glob('DashAI/front/build/**/*', recursive=True))],
    version='0.0.5',
    license='MIT',
    description='DashAI: a graphical toolbox for training, evaluating and deploying state-of-the-art AI models.',
    author='Felipe Bravo-Marquez',
    author_email='fbravo@dcc.uchile.cl',
    url='https://github.com/OpenCENIA/DashAI',
    install_requires=[
        'fastapi[all]',
        'SQLAlchemy',
        'scikit-learn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['dashai'],
)
