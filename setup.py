from setuptools import setup, find_packages

install_requires = [
    "setuptools>=41.0.0",
    "numpy>=1.16.0",
    "joblib",
    "scipy"
]

extras_require = {
    "tf": ["tensorflow==2.0.0"],
    "tf_gpu": ["tensorflow-gpu==2.0.0"]
}

setup(
    name="tf2gan",
    version="0.0.0",
    description="Generative Adversarial Networks using TensorFlow2.x",
    url="https://github.com/keiohta/tf2gan",
    author="Kei Ohta",
    author_email="dev.ohtakei@gmail.com",
    license="MIT",
    packages=find_packages("."),
    install_requires=install_requires,
    extras_require=extras_require)
