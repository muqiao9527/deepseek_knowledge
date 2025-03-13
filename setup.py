from setuptools import setup, find_packages

setup(
    name="knowledge_base_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # 依赖项将从requirements.txt读取
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive knowledge base system",
    keywords="knowledge-base, document-management, search",
    python_requires=">=3.8",
)