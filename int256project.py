!pip install pyswarm

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from pyswarm import pso
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Score history for visualization
scores_history = []

# Define the fitness function for PSO
def fitness_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    criterion_index = int(round(params[3]))  # 0 for 'gini', 1 for 'entropy'
    criterion = ['gini', 'entropy'][criterion_index]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )

    # Use cross-validation to evaluate performance
    score = cross_val_score(model, X, y, cv=3).mean()
    scores_history.append(score)
    return -score  # Because PSO minimizes

# Parameter bounds
# [n_estimators, max_depth, min_samples_split, criterion (0 or 1)]
lb = [10, 3, 2, 0]    # Lower bounds
ub = [200, 20, 20, 1]  # Upper bounds

# Run PSO
best_params, best_score = pso(fitness_function, lb, ub, swarmsize=20, maxiter=30)

# Decode best parameters
best_n_estimators = int(best_params[0])
best_max_depth = int(best_params[1])
best_min_samples_split = int(best_params[2])
best_criterion = ['gini', 'entropy'][int(round(best_params[3]))]

print("\nBest Hyperparameters Found:")
print(f"n_estimators = {best_n_estimators}")
print(f"max_depth = {best_max_depth}")
print(f"min_samples_split = {best_min_samples_split}")
print(f"criterion = {best_criterion}")
print(f"Best CV Accuracy = {-best_score:.4f}")

# Final model training with best parameters
final_model = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    criterion=best_criterion,
    random_state=42
)
final_model.fit(X, y)
final_score = final_model.score(X, y)
print("\nFinal Model Training Accuracy on Full Data:", final_score)

# Plot score history
plt.figure(figsize=(10, 6))
plt.plot(scores_history)
plt.title('PSO Optimization Progress (Cross-Validation Accuracy)')
plt.xlabel('Evaluation Count')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()