import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.title("Isolation Forests Demonstrator")

# Detailed introduction
st.write("""
    Welcome to the Isolation Forests Demonstrator. This app allows you to interactively explore 
    how the Isolation Forest algorithm detects anomalies in a dataset. Adjust the sliders in the sidebar 
    to change the parameters and observe the effects on anomaly detection.
""")

# Sidebar settings with explanations
st.sidebar.header("Isolation Forest Settings")
st.sidebar.markdown("""
    **Number of Samples**  
    Adjust the total number of data points in the synthetic dataset.
""")
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)

st.sidebar.markdown("""
    **Number of Outliers**  
    Set the number of anomalous data points in the dataset.
""")
num_outliers = st.sidebar.slider("Number of Outliers", 1, 50, 10)

st.sidebar.markdown("""
    **Random State**  
    Control the randomness of the data generation for reproducible results.
""")
random_state = st.sidebar.slider("Random State", 1, 42, 42)

st.markdown("""
    **Data Generation**  
    The dataset consists of normally distributed points (inliers) and a smaller number of uniformly distributed 
    points (outliers). The sliders above control the proportion of these points.
""")

# Generate synthetic data
rng = np.random.RandomState(random_state)
X_normal = rng.randn(num_samples - num_outliers, 2)
X_outliers = rng.uniform(low=-4, high=4, size=(num_outliers, 2))
X = np.vstack((X_normal, X_outliers))

# Apply Isolation Forest
clf = IsolationForest(contamination=float(num_outliers) / num_samples, random_state=random_state)
clf.fit(X)
scores_pred = clf.decision_function(X)
labels = clf.predict(X)

# Visualize the data with the anomaly scores
fig, ax = plt.subplots()
colors = np.array(['#0000ff' if l == 1 else '#ff0000' for l in labels])
scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, s=20, edgecolor='k', cmap='viridis')

xx, yy = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500),
                     np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
st.pyplot(fig)

st.markdown("""
    **Model Explanation**  
    The Isolation Forest algorithm isolates anomalies typically with fewer splits compared to normal points. 
    In this visualization, anomalies are points marked in red. The contour line represents the decision 
    boundary of the Isolation Forest: points outside of this line are considered anomalies.
    
    - **Blue points**: Normal observations.
    - **Red points**: Anomalous observations.
    
    The algorithm works by randomly selecting a feature and a split value to isolate points. Anomalies, 
    which are few and different, tend to be isolated quicker, leading to a shorter path in the tree structure, 
    hence a more negative score on the decision function.
""")

# Show the settings used to create the data
st.sidebar.markdown("""
    **Current Settings:**  
    - Number of Samples: {}
    - Number of Outliers: {}
    - Random State: {}
""".format(num_samples, num_outliers, random_state))
