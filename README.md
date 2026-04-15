## QDA Classifiers Node

This directory contains the QDA (Quadratic Discriminant Analysis) classifier node. This node is highly scalable, acting as an independent classification backbone for EEG (power-band) features. Depending on the pipeline configuration, **you can launch multiple instances of this node concurrently** (e.g., one QDA assigned to classify MI, and a separate parallel QDA assigned to CVSA).

---

### 1. Input

* **Topic:** Default is `/eeg_power` but can be dynamically overridden via the `topic_sub` ros parameter.
* **Data:** The node subscribes to this topic. The message structure is defined by the `processing_bci` module. The node expects the `msg.data` field to contain a flattened matrix of EEG signal power, structured as `[bands x channels]`.

---

### 2. Configuration

This Python node **requires mandatory ROS parameters** and a YAML configuration file representing the pre-trained statistical model. 

* `path_qda_model`: The absolute path where the configuration is stored. Models are structurally kept in paradigm-specific sub-folders (e.g., `cfg/mi/qda_model_mi.yaml` and `cfg/cvsa/qda_model_cvsa.yaml`).
* `qda_paradigm`: A string representing the active task (e.g., `mi`, `cvsa`). This governs the naming convention and output topic routing.

The YAML file must contain the following fields:

* `indices`: A list of integers. These are the specific indices from the `[channels x bands]` input matrix that will be used to construct the feature vector.
* `bands`: A list of strings specifying the frequency bands used (e.g., `['delta', 'theta']`).
* `channels`: A list of strings specifying the channels used (e.g., `['F3', 'F4']`).
* `files_used`: A list of strings specifying the original data files used for training.
* `kmeans_model`: The name (or path) of the K-Means YAML model used to generate the labels for this QDA model.
* `n_features`: The total number of features (must match the length of `indices`).
* `n_classes`: The number of classes (e.g., 2).
* `priors`: A list of floats `[n_classes]` representing the prior probabilities for each class.
* `means`: A nested list `[n_classes x n_features]` containing the mean vector for each class.
* `covariance`: A nested list `[n_classes x n_features x n_features]` containing the covariance matrix (or equivalent parameters like scaling/rotation) for each class.

---

### 3. Model Generation

The QDA model (the `.yaml` file) is generated using a **Python script** located in the `/create_qda` directory.

This script is responsible for:
1.  Loading the training dataset. This dataset is located in the `/create_qda/datasets` directory. **This directory is populated by the K-Means node** (see `kmeans_cvsa`), which extracts and saves the EEG data (already filtered for the IC state) and its corresponding class labels into a `.mat` file.
2.  Training the QDA model (using `sklearn` or a similar library) on this data.
3.  Exporting all required QDA parameters (`priors`, `means`, `covariance`, `indices`, etc.) and metadata (`files_used`, `kmeans_model`, `bands`, `channels`) into the YAML format required by this node.

---

### 4. Workflow

1.  **Initialization:** The node identifies its paradigm via `qda_paradigm` and validates the required parameters.
2.  **Load Model:** The node loads the structural QDA parameters (priors, means, covariances, rotations, specific channels/bands) from the defined YAML file.
3.  **Receive Data:** It listens for incoming messages on `/eeg_power` (or the user-defined `topic_sub`).
4.  **Extract Features:** Using the `idchannels` and `bands` from the YAML file, the node accurately reshapes the input array into a `[bands x channels]` matrix and surgically extracts ONLY the relevant indices to build the final feature vector. A log transform (`np.log`) is applied to regularize the powers.
4.  **Classify:** The node calculates the posterior probability for each class using the loaded QDA parameters and assigns the data to the class with the highest probability (Bayesian decision).
5.  **Publish:** The node publishes the resulting classification probability (e.g., the probability of being in the IC state) to the output topic.

---

### 5. Output

* **Topic:** `/{qda_paradigm}/neuroprediction/raw`
* **Message Type:** Publishes both the soft-predictions (posterior probabilities) and a hard-prediction vector inside a `rosneuro_msgs/NeuroOutput` message.

---

### 6. Testing

The validation tests for this node are located in the `test` directory.

The testing process validates the node's functionality by comparing the ROS node's output against a MATLAB simulation using the exact same statistical model and deterministic data structure:

1.  **ROS Execution:** A launch file initiates the benchmark. An auxiliary node publishes pre-defined data (e.g., from a CSV file) to the `/eeg_power` topic. A second helper node subscribes to the dynamic target topic (e.g., `/{qda_paradigm}/neuroprediction/raw`) and dumps the resulting Bayesian probabilities into an output CSV file (e.g., `ros_output.csv`).
2.  **MATLAB Verification:** The output CSV file (`ros_output.csv`) is ingested locally into MATLAB.
3.  **Comparison:** The sequence of probabilities outputted natively by the Python ROS node is mathematically differenced against the results from the equivalent MATLAB algorithm. The MATLAB script uses identical helper functions (also present in the `test` directory) to decode and instantiate the QDA boundary planes from the exact same YAML format. Both platforms process the *same* input sequence.

The benchmark passes if the output probabilities from ROS and MATLAB are identically convergent. **Currently, the absolute accepted error between the discrete implementations is rigorously bound within the order of $10^{-7}$.**