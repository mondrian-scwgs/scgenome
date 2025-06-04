from setuptools import setup, find_packages
import versioneer

setup(
    name='scgenome',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Code for analyzing single cell whole genomes',
    author='Shah Lab',
    url='https://github.com/shahcompbio/scgenome',
    packages=find_packages(),
    install_requires=[
        'anndata',
        'Click',
        'csverve>=0.3.1',
        'hdbscan',
        'matplotlib',
        'nose',
        'numba',
        'numpy==1.23',
        'pandas',
        'pyBigWig',
        'pyfaidx',
        'pyranges',
        'scikit-learn',
        'scipy',
        'seaborn',
        'umap-learn',
    ],
    package_data={
        'scgenome': [
            'data/*',
            'dtypes/*.yaml',
            'datasets/data/*'
        ],
    },
)
