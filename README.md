# Breed-Sense-Image-Based-Cattle-Verification-
An image based recognistion system to identify the breeds of different cattles using SVM.

Breed Sense is a simple machine learning project that helps identify cattle breeds. Instead of relying on guesswork, it analyzes the input and gives predictions based on how confident the model actually is.

The system checks each input and returns up to three possible breeds. But it doesn’t just throw out answers — every prediction has to cross a 75% confidence level. If none of the predictions reach that mark, it clearly says the breed couldn’t be identified.
This makes the output more reliable and avoids misleading results.

How it works:-

Run train.py model which will create a cattle_model.pkl. After the model has completed training run app.py model which provides a local host code to run on browser with the cattle identifier. Install any neccessary library required. 
The model processes the input data, extracts useful features, and runs it through a trained SVM classifier. Based on this, it ranks possible breeds and selects the top three.

Each of these is checked against the confidence threshold:

If confidence ≥ 75% → shown as a valid prediction

If confidence < 75% → ignored

If all predictions are below the threshold, the system simply returns that the breed is not identified.

Why this approach:-

A lot of basic models always give an answer, even when they’re not sure. That leads to wrong results.
Breed Sense is built to avoid that — it only responds when it’s reasonably confident.

Current state:-

The model is trained on a dataset of 500+ samples and is still being improved over time to increase accuracy and support more breeds.

What’s next:-

The plan is to improve the dataset, test better models, and make the system more practical for real-world use.
