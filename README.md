# HEI BA5 Course Summary

## Overview

This repository contains comprehensive summaries, notes, and materials for the HEI Bachelor of Applied Science (BA5) program courses. It serves as a centralized resource for course content, assignments, and study materials to support academic learning and knowledge retention.

## Repository Structure

```
├── 301.1-ml/                    # Machine Learning Course
│   ├── intro.md                 # Introduction to Machine Learning concepts
│   ├── genetic_algorithms.md    # Genetic Algorithms content
│   ├── svm.md                   # Support Vector Machines content
│   ├── pyproject.toml           # Project configuration with uv dependency management
│   ├── uv.lock                  # uv lock file for dependency management
│   ├── .venv/                   # Virtual environment created with uv
│   ├── content/                 # Course content and resources
│   │   ├── week-1/
│   │   │   ├── 1_introduction.pdf
│   │   │   ├── 2_mathematical_foundations.pdf
│   │   │   ├── 3_model_evaluation.pdf
│   │   │   ├── mini_lab_1_fuel_consumption.ipynb   # Jupyter notebook for lab
│   │   │   └── project_1_house.ipynb              # Jupyter notebook for project
│   │   └── week-2/
│   │       ├── Genetic Algorithms.pdf
│   │       └── mastermind_notebook.py              # Python implementation for Mastermind
│   ├── res/                     # Resources directory
│   └── src/                     # Source code directory
├── 302.1-data-computation/      # Data Computation Course
│   └── ...                      # Course content
└── README.md                    # This file
```

## Available Courses

### 301.1 - Machine Learning
- **Content**: Introduction to ML concepts, mathematical foundations, model evaluation, linear regression, genetic algorithms, support vector machines
- **Materials Include**:
  - Weekly summaries (intro.md, genetic_algorithms.md, svm.md)
  - Lecture notes and key concepts
  - Practical labs (Fuel consumption analysis, house price prediction)
  - Projects applying regression models
  - Genetic algorithm implementations
  - Support Vector Machines (SVM) content and examples
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
4. **Set up** the Python environment using uv (if working with Python notebooks): First install uv (https://docs.astral.sh/uv/getting-started/installation/), then run `cd 301.1-ml && uv sync` (uses the provided pyproject.toml)

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