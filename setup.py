import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Auditory_cortex",
    version="0.0.1",
    author="Akshita Ramya and Bilal Ahmed",
    author_email="{akamsali, bahmed}@purdue.edu",
    description="Auditory Cortex Modeling with ECoG data in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akshitark/",
    packages=setuptools.find_packages(),
    package_data={
        'auditory_ctx': [
            'conf/params.yaml'
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pandas', 'sentencepiece', 'transformers'#, 'cupy', 'seaborn', 'plotly', 'naplib', 'scikit-learn' 
    ],
)
