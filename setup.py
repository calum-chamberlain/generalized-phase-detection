from setuptools import setup

setup(
    name='gpd',
    version='0.1',
    description="Pythonic wrapper for Zach Ross' Generalized Phase Detector",
    url=None,
    author='Calum Chamberlain',
    author_email='calum.chamberlain@vuw.ac.nz',
    license='MIT',
    packages=[
        'gpd', 'gpd.helpers', 'gpd.models'],
    zip_safe=False,
    scripts=[],
)
