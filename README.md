# HEI BA5 Course Summary

## Overview

This repository contains comprehensive summaries, notes, and materials for the HEI-VS Bachelor of Computer Science and Communication Systems. It serves as a centralized resource for course content, assignments, and study materials to support academic learning and knowledge retention.

## Repository Structure

```
├── 301.1-ml/                    # Machine Learning Course
│   ├── content/                 # Lecture notes, PDFs, notebooks, and scripts
│   │   ├── week-1/              # Week 1 materials
│   │   └── week-2/              # Week 2 materials
│   ├── res/                     # Generated resources like plots
│   ├── src/                     # Source code for examples
│   └── summaries/               # Course summaries in Markdown
├── 302.1-data-computation/      # Data Computation Course
│   ├── content/                 # Course content including HTML notes
│   ├── quizzes/                 # Quiz scripts
│   └── summaries/               # Course summaries in Markdown
├── pyproject.toml               # Python project configuration
├── uv.lock                      # Dependency lock file
└── README.md                    # This file
```

## Available Courses

### 301.1 - Machine Learning

- **Content**: Introduction to ML concepts, mathematical foundations, model evaluation, linear regression, genetic algorithms, support vector machines, ML pipelines for supervised classification
- **Materials Include**:
  - Weekly summaries (intro.md, ml_pipeline.md, genetic_algorithms.md, svm.md)
  - Lecture notes and key concepts
  - Practical labs (Fuel consumption analysis, house price prediction)
  - Projects applying regression models
  - Genetic algorithm implementations
  - Support Vector Machines (SVM) content and examples
  - ML pipeline workflows with K-NN and SVM models
  - Exam preparation materials
  - Python project configuration with uv dependency management

### 302.1 - Data Computation

- **Content**: Data processing techniques, computational methods, algorithms
- **Materials Include**:
  - Course notes and summaries
  - Practical exercises and assignments
  - Project documentation

## How to Use This Repository

1. **Navigate** to the course directory you're interested in (e.g., `301.1-ml/`)
2. **Explore** the content within each course directory
3. **Review** course materials to support your learning
4. **Set up** the Python environment using uv (if working with Python notebooks, scripts, quizzes, or interactive apps): First install uv (<https://docs.astral.sh/uv/getting-started/installation/>), then run `uv sync` from the repository root (uses the provided pyproject.toml). This creates a virtual environment (.venv) and installs dependencies like streamlit, matplotlib, scikit-learn, marimo, and others as needed.

5. **Run Python-based content**: Use `uv run` to execute scripts, notebooks, quizzes, and apps within the project environment. Examples:
   - Scripts: `uv run python path/to/script.py` (e.g., `uv run python 301.1-ml/src/svm/main.py` to generate SVM plots)
   - Marimo notebooks: `uv run marimo run path/to/notebook.py` (e.g., `uv run marimo run 301.1-ml/content/week-2/mastermind_notebook.py`)
   - Jupyter notebooks: `uv run jupyter notebook path/to/notebooks/` (e.g., `uv run jupyter notebook 301.1-ml/content/`)
   - Streamlit quizzes/apps: `uv run streamlit run path/to/app.py` (e.g., `uv run streamlit run 302.1-data-computation/quizzes/docker_quiz.py` for an interactive Docker quiz with 22 questions, feedback, hints, and scoring)

## Contribution Guidelines

This repository is designed to be a collaborative learning resource. Contributions are welcome in the form of:

- Additional summaries or clarifications of course concepts
- Corrections to existing content
- Additional practice exercises or examples
- Study guides or exam preparation materials
- Improvements to Python implementations and environment setup

To contribute:

1. Fork the repository
2. Create your branch with descriptive naming
3. Make your changes
4. If adding Python dependencies, update the pyproject.toml file accordingly using uv
5. Submit a pull request with an explanation of your contribution

## License

This repository is created for educational purposes within the HEI academic context. All content is either original work or appropriately attributed educational resources.

## Contact

For questions or suggestions regarding this repository, please open an issue in this repository or contact the repository maintainers directly.
