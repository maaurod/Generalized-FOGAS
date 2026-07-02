from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
readme_path = ROOT / "README.md"
requirements_path = ROOT / "requirements.txt"

long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
requirements = [
    line.strip()
    for line in requirements_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="rl-methods",
    version="0.1.0",
    author="Mauro Diaz Lupone",
    author_email="maurodiazlupone@gmail.com",
    description="Reorganized RL methods research codebase",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maaurod/FOGAS",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
