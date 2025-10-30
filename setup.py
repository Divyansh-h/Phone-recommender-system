from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "Phone-recommendation-system"
AUTHOR_USER_NAME = "Advitiyyyaa"
# keep install_requires in sync with requirements.txt
LIST_OF_REQUIREMENTS = ['streamlit', 'numpy', 'scikit-learn']


setup(
    name="Phone-recommendation-system",
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for Mobile Recommendation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="advitiya.arya@gmail.com",
    # Use find_packages() to automatically discover packages. This avoids
    # failing when a hard-coded package directory (like 'src') doesn't exist.
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)