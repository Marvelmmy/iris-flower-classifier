import pickle
import numpy as np

def load_model(modelfile="models/my_model.pkl"):
    with open(modelfile, "rb") as f:
        return pickle.load(f)

def predict_model(model, sample):
    sample = np.array(sample).reshape(1, -1)  # Ensure correct shape
    return model.predict(sample)[0]

if __name__ == "__main__":
    model = load_model()
    
    # Example: if your model expects 4 features (like Iris dataset)
    sample_input = [5.1, 3.5, 1.4, 0.2]  
    
    result = predict_model(model, sample_input)
    print("Predicted class:", result)
