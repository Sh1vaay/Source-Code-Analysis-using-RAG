from setuptools import find_packages, setup

setup(
    name="source-code-analysis-genai",
    version="1.0.0",
    author="CodeRAG Contributors",
    author_email="maintainers@coderag.org",
    description="Production-grade RAG system for source code analysis using LangGraph + Qdrant",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[],
)