# pylint: disable=missing-module-docstring
from setuptools import find_packages
from setuptools import setup

setup(
    name='exception_rules',
    description='''
    Package implementing exception rules.
    ''',
    version='7.5.2',
    author='Dawid Macha',
    author_email='dawid.macha@emag.lukasiewicz.gov.pl',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.0.3',
        'decision_rules==7.5.2',
    ],
    test_suite="tests"
)
