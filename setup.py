# pylint: disable=missing-module-docstring
from setuptools import find_packages
from setuptools import setup

setup(
    name='exception_rules',
    description='''
    Package implementing exception rules.
    ''',
    version='1.0.0',
    author='Dawid Macha',
    author_email='dawid.macha@polsl.pl',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26,<3',
        'pandas>=2.0,<3',
        'pydantic>=2.0,<3',
        'scipy>=1.11,<2',
        'scikit-learn>=1.3,<2',
        'imbalanced-learn>=0.10,<1',
        'typeguard>=4.3,<5',
    ],
    extras_require={
        'test': ['pytest>=8,<9'],
    },
)
