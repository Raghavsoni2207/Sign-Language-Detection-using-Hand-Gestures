import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect the shape of data elements
# for i, element in enumerate(data_dict['data']):
#     print(f"Element {i} has shape: {np.shape(element)}")

# If the shapes are inconsistent, handle them (e.g., padding/truncating/flattening)
# For example, if padding is needed, we can ensure all have the same length
max_length = max([len(x) for x in data_dict['data']])
data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in data_dict['data']])

# Convert labels to NumPy array
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict on test data
y_predict = model.predict(x_test)

# Calculate accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
