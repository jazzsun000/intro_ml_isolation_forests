import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.title("Isolation Forests Demonstrator")

# Sidebar settings
st.sidebar.header("Isolation Forest Settings")
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
num_outliers = st.sidebar.slider("Number of Outliers", 1, 50, 10)
random_state = st.sidebar.slider("Random State", 1, 42, 42)

# Generate synthetic data
rng = np.random.RandomState(random_state)
# Normal data points
X_normal = rng.randn(num_samples - num_outliers, 2)
# Anomaly data points far away from normal ones
X_outliers = rng.uniform(low=-4, high=4, size=(num_outliers, 2))
X = np.vstack((X_normal, X_outliers))

# Apply Isolation Forest
clf = IsolationForest(contamination=float(num_outliers) / num_samples, random_state=random_state)
clf.fit(X)
scores_pred = clf.decision_function(X)
labels = clf.predict(X)

# Visualize the data with the anomaly scores
fig, ax = plt.subplots()
# The color of the points is determined by the anomaly score
colors = np.array(['#0000ff' if l == 1 else '#ff0000' for l in labels])
scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, s=20, edgecolor='k', cmap='viridis')

# Draw a line to separate the anomalies
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 500),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Draw contour lines where the anomaly score is zero (boundary of anomalies)
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
st.pyplot(fig)

st.write("Isolation Forests isolate anomalies typically with fewer splits compared to normal points. In this visualization, "
         "anomalies are points marked in red.")
