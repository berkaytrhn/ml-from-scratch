# Regularization Tests

## Linear Regression

| Hyperparameter | Value | 
|----------------|---------------|
| Epochs | 10_000 | 
| Learning Rate | 0.0001 | 
| Lambda | 1e-4 | 


| Regularization | Training Loss | Test Loss |
|----------------|---------------|-----------|
| Without Regularization | 1.014**2896** | 1.0448**390** |
| With L1 Regularization | 1.014**5463** | 1.0448**370** |
| With L2 Regularization | 1.014**3669** | 1.0448**407** |

<br>

## Logistic Regression


| Hyperparameter | Value | 
|----------------|---------------|
| Epochs | 10_000 | 
| Learning Rate | 0.001 | 
| Lambda | 1e-4 | 



| Regularization | Train Loss |  Training Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 Score | 
|----------------|---------------|---------------|-----------|-----------|-----------|-----------|
| Without Regularization | 0.1720 | 95.0131% | 93.6170% | 0.9360 | 0.9669 | 0.9512 |
| With L1 Regularization | 0.1730 | 95.0131% | 93.6170% | 0.9360 | 0.9669 | 0.9512 |
| With L2 Regularization | 0.1723 | 95.0131% | 93.6170% | 0.9360 | 0.9669 | 0.9512 |

- Not meaningful change probably due to simplicity of the dataset