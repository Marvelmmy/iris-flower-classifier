from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import seaborn as sn
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model with test data and show confusion matrix."""
    
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualization
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy, report, cm
