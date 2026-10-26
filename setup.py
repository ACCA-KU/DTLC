from setuptools import setup, find_packages

setup(
    name="SEI",
    version="0.1",
    packages=find_packages(),
    install_requires=[
"numpy",
"pandas",
"scikit-learn",
"scipy",
"matplotlib",
"seaborn",
"torch>=2.2,<=2.4.0",
"rdkit>=2022",
"tqdm",
"joblib",
"pydantic",
"dgl",
#"MoleculeSculptor" --> 그냥 D4CMPP_TLC 내부에 로컬 파일로 넣어둠
    ],
    python_requires='>=3.8',
)