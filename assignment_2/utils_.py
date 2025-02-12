import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from typing import List

"""
References:
    [https://github.com/udlbook/udlbook/blob/main/Notebooks/Chap06/]
"""
def plot_data(x: np.ndarray, y: np.ndarray, title: str = "Data"):
    """
    Plot the data.
    Args:
        x: x values
        y: y values
        title: title of the plot
    
    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.plot(x, y, 'bo')
    ax.set_xlabel('Input, $x$'); ax.set_ylabel('Output, $y$')
    if title is not None:
        ax.set_title(f"{title}")
    # plt.grid()
    plt.show()

def plot_loss_curves(title: str = "Loss Curves", loss_histories: list = None, model_names: list = None):
    """
    Plot the loss curves.
    Args:
        loss_histories: list of loss_history for each model
        title: title of the plot
    
    Returns:
        None
    """
    fig, ax = plt.subplots()
    markers = ['r', 'g', 'y', 'm']
    for loss_history, marker, model_name in zip(loss_histories, markers, model_names):
        ax.plot(loss_history, c=marker, label=model_name)
    
    ax.set_xlabel('Epoch, $t$'); ax.set_ylabel('Loss, $J$')
    if title is not None:
        ax.set_title(f"{title}")
    
    plt.legend()
    plt.grid(True, 'major')
    plt.show()

def draw_models(models: List[object]):
    """
    Draw the data and the model.
    Args:
        model: function that takes x and returns y
        title: title of the plot
    
    Returns:
        None
    """
    fig, ax = plt.subplots(1, len(models), figsize=(20, 4))
    markers = ['r', 'g', 'y', 'm']
    i = 0
    for model, marker in zip(models, markers):
        x_model = np.arange(np.min(model.data_x), np.max(model.data_x), 0.001)
        y_model = model.model(x_model)
        cost = model.compute_cost(model.data_x, model.data_y)

        ax[i].plot(model.data_x, model.data_y,'bo')
        ax[i].plot(x_model, y_model, marker, linewidth=2)
        # ax[i].set_xlabel('x'); ax[i].set_ylabel('y')
        ax[i].set_title(f"{model.varient}, Loss: ${cost:.2f}$")
        i += 1
    fig.text(0.5, 0.00, 'Input, $x$', ha='center')
    for ax_row in ax:
        ax_row.set_ylabel('Output, $y$')
    plt.tight_layout()
    plt.show()


def draw_loss_function(base_model: object, theta_iters: bool = False, theta_histories: list = None, model_names: list = None):
    """
    Draw the loss function of the model.
    Args:
        base_model: model object
        theta_iters: whether to plot the iterations of theta
        theta_histories: list of lists containing iterations of theta
        model_names: vareients of these models.
    Returns:
        None
    """
    # Define pretty colormap
    my_colormap_vals_hex =('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', '320f0a', '330f0b', '34100b', '35110c', '36110d', '37120e', '38120f', '39130f', '3a1410', '3b1411', '3c1511', '3d1612', '3e1613', '3f1713', '401714', '411814', '421915', '431915', '451a16', '461b16', '471b17', '481c17', '491d18', '4a1d18', '4b1e19', '4c1f19', '4d1f1a', '4e201b', '50211b', '51211c', '52221c', '53231d', '54231d', '55241e', '56251e', '57261f', '58261f', '592720', '5b2821', '5c2821', '5d2922', '5e2a22', '5f2b23', '602b23', '612c24', '622d25', '632e25', '652e26', '662f26', '673027', '683027', '693128', '6a3229', '6b3329', '6c342a', '6d342a', '6f352b', '70362c', '71372c', '72372d', '73382e', '74392e', '753a2f', '763a2f', '773b30', '783c31', '7a3d31', '7b3e32', '7c3e33', '7d3f33', '7e4034', '7f4134', '804235', '814236', '824336', '834437', '854538', '864638', '874739', '88473a', '89483a', '8a493b', '8b4a3c', '8c4b3c', '8d4c3d', '8e4c3e', '8f4d3f', '904e3f', '924f40', '935041', '945141', '955242', '965343', '975343', '985444', '995545', '9a5646', '9b5746', '9c5847', '9d5948', '9e5a49', '9f5a49', 'a05b4a', 'a15c4b', 'a35d4b', 'a45e4c', 'a55f4d', 'a6604e', 'a7614e', 'a8624f', 'a96350', 'aa6451', 'ab6552', 'ac6552', 'ad6653', 'ae6754', 'af6855', 'b06955', 'b16a56', 'b26b57', 'b36c58', 'b46d59', 'b56e59', 'b66f5a', 'b7705b', 'b8715c', 'b9725d', 'ba735d', 'bb745e', 'bc755f', 'bd7660', 'be7761', 'bf7862', 'c07962', 'c17a63', 'c27b64', 'c27c65', 'c37d66', 'c47e67', 'c57f68', 'c68068', 'c78169', 'c8826a', 'c9836b', 'ca846c', 'cb856d', 'cc866e', 'cd876f', 'ce886f', 'ce8970', 'cf8a71', 'd08b72', 'd18c73', 'd28d74', 'd38e75', 'd48f76', 'd59077', 'd59178', 'd69279', 'd7937a', 'd8957b', 'd9967b', 'da977c', 'da987d', 'db997e', 'dc9a7f', 'dd9b80', 'de9c81', 'de9d82', 'df9e83', 'e09f84', 'e1a185', 'e2a286', 'e2a387', 'e3a488', 'e4a589', 'e5a68a', 'e5a78b', 'e6a88c', 'e7aa8d', 'e7ab8e', 'e8ac8f', 'e9ad90', 'eaae91', 'eaaf92', 'ebb093', 'ecb295', 'ecb396', 'edb497', 'eeb598', 'eeb699', 'efb79a', 'efb99b', 'f0ba9c', 'f1bb9d', 'f1bc9e', 'f2bd9f', 'f2bfa1', 'f3c0a2', 'f3c1a3', 'f4c2a4', 'f5c3a5', 'f5c5a6', 'f6c6a7', 'f6c7a8', 'f7c8aa', 'f7c9ab', 'f8cbac', 'f8ccad', 'f8cdae', 'f9ceb0', 'f9d0b1', 'fad1b2', 'fad2b3', 'fbd3b4', 'fbd5b6', 'fbd6b7', 'fcd7b8', 'fcd8b9', 'fcdaba', 'fddbbc', 'fddcbd', 'fddebe', 'fddfbf', 'fee0c1', 'fee1c2', 'fee3c3', 'fee4c5', 'ffe5c6', 'ffe7c7', 'ffe8c9', 'ffe9ca', 'ffebcb', 'ffeccd', 'ffedce', 'ffefcf', 'fff0d1', 'fff2d2', 'fff3d3', 'fff4d5', 'fff6d6', 'fff7d8', 'fff8d9', 'fffada', 'fffbdc', 'fffcdd', 'fffedf', 'ffffe0')
    my_colormap_vals_dec = np.array([int(element, base=16) for element in my_colormap_vals_hex])
    r = np.floor(my_colormap_vals_dec/(256 * 256))
    g = np.floor((my_colormap_vals_dec - r * 256 * 256) / 256)
    b = np.floor(my_colormap_vals_dec - r * 256 * 256 - g * 256)
    my_colormap = ListedColormap(np.vstack((r, g, b)).transpose() / 255.0)

    # Make grid of values to plot
    if base_model.name == "LinearRegression":
        theta_0_mesh, theta_1_mesh = np.meshgrid(np.arange(0.0, 2.0, 0.02), np.arange(-0.5, 1.5, 0.002))
        model_fn = lambda x, theta: theta[0] + theta[1] * x
    elif base_model.name == "Gabor":
        theta_0_mesh, theta_1_mesh = np.meshgrid(np.arange(-3.5, 2.5, 0.1), np.arange(12.0, 25.0, 0.1))
        model_fn = lambda x, theta: np.sin(theta[0] + 0.06 * theta[1] * x) * np.exp(-0.5 * (x - 5)**2 / theta[1])

    loss_fn = lambda x, y, theta: np.sum((y - model_fn(x, theta))**2)
    loss_mesh = np.zeros_like(theta_0_mesh)
    
    # Compute loss for every set of parameters
    for id_theta_1, theta_1 in np.ndenumerate(theta_1_mesh):
        loss_mesh[id_theta_1] = loss_fn(base_model.data_x, base_model.data_y, np.array([[theta_0_mesh[id_theta_1]], [theta_1]]))

    fig,ax = plt.subplots()
    fig.set_size_inches(8, 6.4)
    contour = ax.contourf(theta_0_mesh, theta_1_mesh, loss_mesh, 256, cmap=my_colormap)
    ax.contour(theta_0_mesh, theta_1_mesh, loss_mesh, 40, colors=['#80808080'])

    if theta_iters is not False and theta_histories is not None:
        markers = ['ro-', 'go-', 'yo-', 'mo-']
        for model_name, theta_history, marker in zip(model_names, theta_histories, markers[:len(model_names)]):
            ax.plot(theta_history[0], theta_history[1], marker, label=model_name, markersize=0.5)
        ax.legend(loc='lower left')
    
    if base_model.name == "LinearRegression":
        ax.set_ylim([1.5, -0.5])
        ax.set_xlabel('$Intercept, θ_0$'); ax.set_ylabel('$Slope, θ_1$')

    elif base_model.name == "Gabor":
        ax.set_ylim([12.0, 25.0])
        ax.set_xlabel('$Offset, θ_0$'); ax.set_ylabel('$Frequency, θ_1$')

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Loss, $J$')
    plt.show()

def plot_auto_reduce(fn, initial_lr: float = 0.01, eta: float = 0.9, max_epochs: int = 30, n_events: int = 5):
    """
    Plot the auto reduce function.
    Args:
        fn: function that takes x and returns y
    Returns:
        None
    """
    x = np.arange(0, max_epochs)
    y = np.array([fn(initial_lr, eta, x_i, max_epochs, n_events) for x_i in x])
    fig, ax = plt.subplots()
    ax.step(x, y, 'b')
    ax.set_xlabel('Epoch, $t$'); ax.set_ylabel('Learning rate, $\\alpha(t)$')
    plt.title('Auto Reduce Function')
    plt.grid(True, 'major')
    plt.show()

def plot_poly_decay(fn, initial_lr: float = 0.01, final_lr: float = 0.001, eta: float = 0.5, max_epochs: int = 30):
    """
    Plot the poly decay function.
    Args:
        fn: function that takes x and returns y
    Returns:
        None
    """
    x = np.arange(0, max_epochs)
    y = np.array([fn(initial_lr, final_lr, x_i, max_epochs, eta) for x_i in x])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b')
    ax.set_xlabel('Epoch, $t$'); ax.set_ylabel('Learning rate, $\\alpha(t)$')
    plt.title('Polynomial Decay Function')
    plt.grid(True, 'major')
    plt.show()
