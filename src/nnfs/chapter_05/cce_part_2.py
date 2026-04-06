import numpy as np
softmax_outputs = np.array([
                            [0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]
                            ])
class_targets = np.array([
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]
                        ])

# Probabilities for target values - 
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]

# Mak values - only for one-hot encoded lables
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs*class_targets,
        axis=1
    )

# losses
neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)
print(average_loss)

print(np.e**(-np.inf))

# Error Expected
# print(np.mean([1, 2, 3,-np.log(0)]))
# Adding insignifacnce to correct and showing edge cases that are now problems
print(-np.log(1e-7))
print(-np.log(1+1e-7))
print(-np.log(1-1e-7))
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
