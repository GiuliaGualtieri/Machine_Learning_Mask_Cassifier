from setuptools import setup

setup(
    name='machine_learning_mask_classifier',
    version='0.0.1',
    packages=find_packages(),
    description='Python repository to do practice in french',
    license='GPL-3.0 license',
    install_requires=[
    "joblib",
    "numpy",
    "opencv-python",
    "Pillow",
    "scikit-learn",
    "scipy"
    "threadpoolctl"],
)