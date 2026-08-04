# ChaosEngine

> An Interactive Machine Learning Experimentation Platform for Predictive Maintenance

---

# Vision

ChaosEngine is being built as an educational yet production-inspired ML platform.

Instead of implementing isolated machine learning algorithms, the objective is to create a reusable platform where different algorithms can be trained, evaluated, compared and deployed on the same dataset.

Every algorithm studied throughout this journey will be integrated into the platform.

---

# Current Progress

## Phase 1 — Platform Architecture ✅

Completed:

- Repository restructuring
- Modular folder architecture
- GitHub project setup
- Planning for reusable ML pipeline

---

## Software Engineering Concepts Learned

### Separation of Concerns

Every component should have one clearly defined responsibility.

Example:

- Loader → Load datasets
- Preprocessing → Clean and transform data
- Models → Train ML algorithms
- Evaluation → Compare model performance

---

### Single Responsibility Principle (SRP)

Each module should have one reason to change.

Examples:

- loader.py changes only if data loading changes.
- preprocessing.py changes only if preprocessing changes.
- trainer.py changes only if training workflow changes.

---

### Reusability

Reusable modules reduce duplicated code.

Instead of reading CSV files inside every algorithm,

everything calls

load_dataset()

---

### DRY (Don't Repeat Yourself)

Shared logic should exist in only one place.

Examples:

- Dataset loading
- Data normalization
- Missing value handling

---

### Abstraction

Modules expose only what other modules need.

Instead of exposing

pd.read_csv(...)

the loader exposes

load_dataset()

Future data sources (CSV, SQL, S3) can be added without affecting other modules.

---

### Encapsulation

Implementation details remain hidden.

The models only know they receive a DataFrame.

They do not know where it came from.

---

### Low Coupling

Modules should know as little as possible about each other.

Changing the dataset source should only affect loader.py.

---

### High Cohesion

Each module contains closely related functionality.

loader.py

- load_dataset()
- show_info()
- show_shape()

Nothing unrelated.

---

### Dependency Injection

Modules should receive dependencies instead of creating them.

Example:

Trainer(model)

instead of

Trainer() creating LinearRegression internally.

---

### Interface-based Design

Every model will implement the same interface.

train()

predict()

evaluate()

save()

load()

This allows the platform to swap algorithms without changing other modules.

---

## Current Architecture

```
   Dataset
      │
      ▼
    Loader
      │
      ▼
 Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Factory
      │
      ▼
Machine Learning Model
      │
      ▼
  Evaluation
      │
      ▼
  Dashboard
```

---

# Future Roadmap

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- Support Vector Machines
- Explainable AI (SHAP)
- FastAPI
- Docker
- CI/CD
- GitHub Actions
- MLflow
- Deployment

---

# Learning Philosophy

This project is not intended to demonstrate the use of machine learning libraries alone.

Instead, it focuses on understanding:

- Why the architecture exists
- Why every component has a single responsibility
- How different ML algorithms fit into a reusable platform
- How production ML systems are designed

Every new algorithm added to ChaosEngine will follow the same architecture and integrate into the existing pipeline without major modifications.
