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
        "ipykernel",
        "ipywidgets",
        "toolz",
        "h5py",
        "joblib",
        "scikit-learn",
        "umap-learn",
        "colorcet",
        "click",
        "multiprocess",
        "opencv-python-headless",
        "ruamel.yaml",
        "statsmodels",
        "polars",
        # "git+https://github.com/wingillis/jax-moseq.git@robust_arhmm"
    ],
    extras_require={
        "jax": ["jax", "optax", "tensorflow-probability"],
        "torch": ["torch", "kornia", "lightning", "torchvision"],
        "all": ["jax", "optax", "torch", "kornia", "lightning", "tensorflow-probability", "torchvision"],
    },
)
