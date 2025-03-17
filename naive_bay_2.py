#Naive bayes calssifier
# Import stuff we will be needing
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import datasets, model_selection

# Load the dataset
iris = datasets.load_iris()

# Use only the first two features: sepal length and width
data = iris.data[:, :2]
targets = iris.target

# Randomly shuffle the data and make train and test splits
x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(data, targets, test_size=0.2)

# Plot the training data
labels = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
label_colours = ['blue', 'red', 'green']

def plot_data(x, y, labels, colours):
    for y_class in np.unique(y):
        index = np.where(y == y_class)
        plt.scatter(x[index, 0], x[index, 1],
                    label=labels[y_class], c=colours[y_class])
    plt.title("Training set")
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.legend()
    
plt.figure(figsize=(8, 5))
plot_data(x_train, y_train, labels, label_colours)
plt.show()
def learn_parameters(x, y, mus, scales, optimiser, epochs):
    """
    Set up the class conditional distributions as a MultivariateNormalDiag
    object, and update the trainable variables in a custom training loop.
    """
    @tf.function
    def nll(dist, x_train, y_train):
        log_probs = dist.log_prob(x_train)
        L = len(tf.unique(y_train)[0])
        y_train = tf.one_hot(indices=y_train, depth=L)
        return -tf.reduce_mean(log_probs * y_train)

    @tf.function
    def get_loss_and_grads(dist, x_train, y_train):
        with tf.GradientTape() as tape:
            tape.watch(dist.trainable_variables)
            loss = nll(dist, x_train, y_train)
            grads = tape.gradient(loss, dist.trainable_variables)
        return loss, grads

    nll_loss = []
    mu_values = []
    scales_values = []
    x = tf.cast(np.expand_dims(x, axis=1), tf.float32)
    dist = tfd.MultivariateNormalDiag(loc=mus, scale_diag=scales)
    for epoch in range(epochs):
        loss, grads = get_loss_and_grads(dist, x, y)
        optimiser.apply_gradients(zip(grads, dist.trainable_variables))
        nll_loss.append(loss)
        mu_values.append(mus.numpy())
        scales_values.append(scales.numpy())
    nll_loss, mu_values, scales_values = \
        np.array(nll_loss), np.array(mu_values), np.array(scales_values)
    return (nll_loss, mu_values, scales_values, dist)
    # Assign initial values for the model's parameters
mus = tf.Variable([[1., 1.], [1., 1.], [1., 1.]])
scales = tf.Variable([[1., 1.], [1., 1.], [1., 1.]])
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
epochs = 10000
nlls, mu_arr, scales_arr, class_conditionals = \
    learn_parameters(x_train, y_train, mus, scales, opt, epochs)
    # Plot the loss and convergence of the standard deviation parameters
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(nlls)
ax[0].set_title("Loss vs. epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Negative log-likelihood")
for k in [0, 1, 2]:
    ax[1].plot(mu_arr[:, k, 0])
    ax[1].plot(mu_arr[:, k, 1])
ax[1].set_title("ML estimates for model's\nmeans vs. epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Means")
for k in [0, 1, 2]:
    ax[2].plot(scales_arr[:, k, 0])
    ax[2].plot(scales_arr[:, k, 1])
ax[2].set_title("ML estimates for model's\nscales vs. epoch")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Scales")
plt.show()
# View the distribution parameters
print("Class conditional means:")
print(class_conditionals.loc.numpy())
print("\nClass conditional standard deviations:")
print(class_conditionals.stddev().numpy())
def get_prior(y):
    """
    This function takes training labels as a numpy array y of shape (num_samples,) as an input,
    and builds a Categorical Distribution object with empty batch shape and event shape,
    with the probability of each class.
    """
    counts = np.bincount(y)
    dist = tfd.Categorical(probs=counts/len(y))
    return dist

prior = get_prior(y_train)
prior.probs
def predict_class(prior, class_conditionals, x):
    def predict_fn(myx):
        class_probs = class_conditionals.prob(tf.cast(myx, dtype=tf.float32))
        prior_probs = tf.cast(prior.probs, dtype=tf.float32)
        class_times_prior_probs = class_probs * prior_probs
        Q = tf.reduce_sum(class_times_prior_probs)       # Technically, this step
        P = tf.math.divide(class_times_prior_probs, Q)   # and this one, are not necessary.
        Y = tf.cast(tf.argmax(P), dtype=tf.float64)
        return Y
    y = tf.map_fn(predict_fn, x)
    return y
