from setuptools import setup, find_packages

def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
    name='mlprotection',
    version='0.1.3',
    author='ivblz',
    description='Библиотека для обнаружения аномалий и потенциально "отравленных" данных в датасетах',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/ivblz/mlprotection',
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0"
    ],
    project_urls={
        'Documentation': 'https://github.com/ivblz/mlprotection'
    },
    python_requires='>=3.6'
)