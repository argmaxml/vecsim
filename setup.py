from distutils.core import setup

__package__= "vecsim"
__version__=""
with open(__package__+"/__init__.py", 'r') as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line)
            break
setup(
    name=__package__,
    packages=[__package__],
    install_requires=[
        "numpy>=1.21.2",
        "pandas>=1.3.0",
        "scikit-learn>=0.19.0",
    ],
    long_description="https://github.com/argmaxml/vecsim/blob/master/README.md",
    long_description_content_type="text/markdown",
    version=__version__,
    description='',
    author='ArgmaxML',
    author_email='ugoren@argmax.ml',
    url='https://github.com/argmaxml/vecsim',
    keywords=['vector-similarity','faiss','hnsw','redis','matching','ranking'],
    classifiers=[],
    extras_require = {
        'faiss': ['faiss-cpu>=1.7.1'],
        'hnsw': ['hnswlib>=0.5.1'],
        'redis': ['redis>=4.3.0'],
        's3': ['smart_open[s3]~=3.0.0'],
        'postgres': ['psycopg2-binary~=2.9.3',"SQLAlchemy~=1.3.22"],

    }
)