from setuptools import setup, find_packages

setup(
    name="aging",
    version="0.0.1",
    author="Winthrop Gillis, Dana Rubi Levy",
    author_email="win.gillis@gmail.com",
    packages=find_packages(),
    python_requires=">3.7",
    install_requires=[
        "pandas",
        "pyarrow",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "torch",
        "ipykernel",
        "ipywidgets",
        "toolz",
        "h5py",
        "joblib",
        "scikit-learn",
        "umap-learn",
        "colorcet",
        "kornia",
        "lightning",
        "click",
        "multiprocess",
        "optax",
        "jax",
        "tensorflow-probability",
        "statsmodels",
        #"git+https://github.com/wingillis/jax-moseq.git@robust_arhmm"
    ]
)
