import networkx as nx
import matplotlib.pyplot as plt
import cupy as cp
from scipy.spatial import ConvexHull
from matplotlib.animation import ArtistAnimation

def draw_neural_web(W, ax, G, return_objs=False):
    """
    Visualizes a neural web by drawing the neural network structure.

    Args:
        W : numpy.ndarray
            A 2D array representing the connection weights of the neural network.
        ax : matplotlib.axes.Axes
            The matplotlib axes where the graph will be drawn.
        G : networkx.Graph
            The NetworkX graph representing the neural network structure.
        return_objs : bool, optional
            If True, returns the drawn objects (nodes and edges). Default is False.

    Returns:
        art1 : matplotlib.collections.PathCollection or None
            Returns the node collection if return_objs is True; otherwise, returns None.
        art2 : matplotlib.collections.LineCollection or None
            Returns the edge collection if return_objs is True; otherwise, returns None.
        art3 : matplotlib.collections.TextCollection or None
            Returns the label collection if return_objs is True; otherwise, returns None.

    Example:
        art1, art2, art3 = draw_neural_web(W, ax, G, return_objs=True)
        plt.show()
    """
    W = W.get()
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] != 0:
                G.add_edge(f'Output{i}', f'Input{j}', ltpw=W[i, j])

    edges = G.edges(data=True)
    weights = [edata['ltpw'] for _, _, edata in edges]
    pos = {}

    num_motor_neurons = W.shape[0]
    num_sensory_neurons = W.shape[1]

    for j in range(num_sensory_neurons):
        pos[f'Input{j}'] = (0, j)

    motor_y_start = (num_sensory_neurons - num_motor_neurons) / 2
    for i in range(num_motor_neurons):
        pos[f'Output{i}'] = (1, motor_y_start + i) 


    art1 = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000, node_color='lightblue')
    art2 = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights, edge_cmap=plt.cm.Blues, width=2)
    art3 = nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Neural Web')

    if return_objs == True:

        return art1, art2, art3


def draw_model_architecture(model_name, model_path=''):
    """
    The `draw_model_architecture` function visualizes the architecture of a neural network model with
    multiple inputs based on activation functions.
    
    :param model_name: The `model_name` parameter in the `draw_model_architecture` function is used to
    specify the name of the neural network model whose architecture you want to visualize. This function
    visualizes the architecture of a neural network model with multiple inputs based on activation
    functions
    :param model_path: The `model_path` parameter in the `draw_model_architecture` function is used to
    specify the path where the neural network model is saved. If the model is saved in a specific
    directory or file location, you can provide that path as a string when calling the function. If the
    model is saved
    """
    """
    Visualizes the architecture of a neural network model with multiple inputs based on activation functions.
    """
    
    from ..model_ops import load_model
    
    model = load_model(model_name=model_name, model_path=model_path)
    
    W = model.weights
    activations = model.activations
    scaler_params = model.scaler_params
    
    # Calculate dimensions based on number of activation functions
    num_activations = len(activations)
    input_groups = num_activations  # Number of input groups equals number of activations
    num_inputs = W.shape[1]

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Calculate positions for nodes
    def get_node_positions():
        positions = {}
        
        # Input layer positions
        total_height = 0.8  # Maksimum dikey alan
        group_height = total_height / input_groups  # Her grup için ayrılan dikey alan
        input_spacing = min(group_height / (num_inputs + 1), 0.1)  # Her girdi arasındaki mesafe
        
        for group in range(input_groups):
            group_start_y = 0.9 - (group * group_height)  # Grubun başlangıç y koordinatı
            for i in range(num_inputs):
                y_pos = group_start_y - ((i + 1) * input_spacing)
                positions[f'input_{group}_{i}'] = (0.2, y_pos)
        
        # Aggregation layer positions
        agg_spacing = total_height / (num_inputs + 1)
        for i in range(num_inputs):
            positions[f'summed_{i}'] = (0.5, 0.9 - ((i + 1) * agg_spacing))
        
        # Output layer positions
        output_spacing = total_height / (W.shape[0] + 1)
        for i in range(W.shape[0]):
            positions[f'output_{i}'] = (0.8, 0.9 - ((i + 1) * output_spacing))
        
        return positions
    
    # Draw the network
    pos = get_node_positions()
    
    # Draw nodes
    for group in range(input_groups):
        # Draw input nodes
        for i in range(num_inputs):
            plt.plot(*pos[f'input_{group}_{i}'], 'o', color='lightgreen', markersize=20)
            plt.text(pos[f'input_{group}_{i}'][0] - 0.05, pos[f'input_{group}_{i}'][1], 
                    f'Input #{i+1} ({activations[group]})', ha='right', va='center')
            
            # Draw connections from input to summed input directly
            plt.plot([pos[f'input_{group}_{i}'][0], pos[f'summed_{i}'][0]],
                    [pos[f'input_{group}_{i}'][1], pos[f'summed_{i}'][1]], 'k-')
            # Draw aggregation nodes
            if group == 0:
                plt.plot(*pos[f'summed_{i}'], 'o', color='lightgreen', markersize=20)
                plt.text(pos[f'summed_{i}'][0], pos[f'summed_{i}'][1] + 0.02,
                        f'Summed\nInput #{i+1}', ha='center', va='bottom')
    
    # Draw output nodes and connections
    for i in range(W.shape[0]):
        plt.plot(*pos[f'output_{i}'], 'o', color='gold', markersize=20)
        plt.text(pos[f'output_{i}'][0] + 0.05, pos[f'output_{i}'][1],
                f'Output #{i+1}', ha='left', va='center', color='purple')
        
        # Connect all aggregation nodes to each output
        for group in range(num_inputs):
            plt.plot([pos[f'summed_{group}'][0], pos[f'output_{i}'][0]],
                    [pos[f'summed_{group}'][1], pos[f'output_{i}'][1]], 'k-')
    
    # Add labels and annotations
    plt.text(0.2, 0.95, 'Input Layer', ha='center', va='bottom', fontsize=12)
    plt.text(0.5, 0.95, 'Aggregation\nLayer', ha='center', va='bottom', fontsize=12)
    plt.text(0.8, 0.95, 'Output Layer', ha='center', va='bottom', fontsize=12)
    
    # Remove axes
    plt.axis('off')
    
    # Add model information
    if scaler_params is None:
        plt.text(0.95, 0.05, 'Standard Scaler=No', fontsize=10, ha='right', va='bottom')
    else:
        plt.text(0.95, 0.05, 'Standard Scaler=Yes', fontsize=10, ha='right', va='bottom')

    # Add model architecture title
    plt.text(0.95, 0.1, f"PLAN Model Architecture: {model_name}", fontsize=12, ha='right', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
def draw_activations(x_train, activation):

    from . import activation_functions as af

    if activation == 'sigmoid':
        result = af.Sigmoid(x_train)

    elif activation == 'circular':
        result = af.circular_activation(x_train)

    elif activation == 'mod_circular':
        result = af.modular_circular_activation(x_train)

    elif activation == 'tanh_circular':
        result = af.tanh_circular_activation(x_train)

    elif activation == 'leaky_relu':
        result = af.leaky_relu(x_train)

    elif activation == 'relu':
        result = af.Relu(x_train)

    elif activation == 'softmax':
        result = af.Softmax(x_train)

    elif activation == 'tanh':
        result = af.tanh(x_train)

    elif activation == 'sinakt':
        result = af.sinakt(x_train)

    elif activation == 'p_squared':
        result = af.p_squared(x_train)

    elif activation == 'sglu':
        result = af.sglu(x_train, alpha=1.0)

    elif activation == 'dlrelu':
        result = af.dlrelu(x_train)

    elif activation == 'sin_plus':
        result = af.sin_plus(x_train)

    elif activation == 'acos':
        result = af.acos(x_train, alpha=1.0, beta=0.0)

    elif activation == 'isra':
        result = af.isra(x_train)

    elif activation == 'waveakt':
        result = af.waveakt(x_train)

    elif activation == 'arctan':
        result = af.arctan(x_train)

    elif activation == 'bent_identity':
        result = af.bent_identity(x_train)

    elif activation == 'softsign':
        result = af.softsign(x_train)

    elif activation == 'pwl':
        result = af.pwl(x_train)

    elif activation == 'sine':
        result = af.sine(x_train)

    elif activation == 'tanh_square':
        result = af.tanh_square(x_train)

    elif activation == 'linear':
        result = x_train

    elif activation == 'sine_square':
        result = af.sine_square(x_train)

    elif activation == 'logarithmic':
        result = af.logarithmic(x_train)

    elif activation == 'sine_offset':
        result = af.sine_offset(x_train, 1.0)

    elif activation == 'spiral':
        result = af.spiral_activation(x_train)

    try: return result
    except:
        print('\rWARNING: error in drawing some activation.', end='')
        return x_train

        
def plot_decision_space(x, y, y_preds=None, s=100, color='tab20'):
    
    from .metrics import pca
    from .data_ops import decode_one_hot
    
    if x.shape[1] > 2:

        X_pca = pca(x, n_components=2)
    else:
        X_pca = x

    if y_preds == None:
        y_preds = decode_one_hot(y)

    y = decode_one_hot(y)
    num_classes = len(cp.unique(y))
    
    cmap = plt.get_cmap(color)


    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', s=50, cmap=cmap, norm=norm)
    

    for cls in range(num_classes):

        class_points = []


        for i in range(len(y)):
            if y_preds[i] == cls:
                class_points.append(X_pca[i])
        
        class_points = cp.array(class_points, dtype=y.dtype)
        

        if len(class_points) > 2:
            hull = ConvexHull(class_points)
            hull_points = class_points[hull.vertices]

            hull_points = cp.vstack([hull_points, hull_points[0]])
            
            plt.fill(hull_points[:, 0], hull_points[:, 1], color=cmap(norm(cls)), alpha=0.3, edgecolor='k', label=f'Class {cls} Hull')

    plt.title("Decision Space (Data Distribution)")

    plt.draw()
    
        
def update_neuron_history(LTPW, ax1, row, col, class_count, artist5, fig1, acc=False, loss=False):

    for j in range(class_count):
        
            if acc != False and loss != False:
                suptitle_info = ' Accuracy:' + str(acc) + '\n' + '\nNeurons Memory:'
            else:
                suptitle_info = 'Neurons Memory:'

            mat = LTPW[j,:].reshape(row, col).get()

            title_info = f'{j+1}. Neuron'
            
            art5 = ax1[j].imshow(mat, interpolation='sinc', cmap='viridis')

            ax1[j].set_aspect('equal')
            ax1[j].set_xticks([])
            ax1[j].set_yticks([])
            ax1[j].set_title(title_info)

           
            artist5.append([art5])

    fig1.suptitle(suptitle_info, fontsize=16)

""" DISABLED
def initialize_visualization_for_fit(val, show_training, neurons_history, x_train, y_train):
    from .data_operations_cuda import find_closest_factors
    visualization_objects = {}

    if show_training or neurons_history:
        if not val:
            raise ValueError("For showing training or neurons history, 'val' parameter must be True.")

        G = nx.Graph()
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Train History')
        visualization_objects.update({
            'G': G,
            'fig': fig,
            'ax': ax,
            'artist1': [],
            'artist2': [],
            'artist3': [],
            'artist4': []
        })

    if neurons_history:
        row, col = find_closest_factors(len(x_train[0]))
        fig1, ax1 = plt.subplots(1, len(y_train[0]), figsize=(18, 14))
        visualization_objects.update({
            'fig1': fig1,
            'ax1': ax1,
            'artist5': [],
            'row': row,
            'col': col
        })

    return visualization_objects
"""
    

""" DISABLED
def update_weight_visualization_for_fit(ax, LTPW, artist2):
    art2 = ax.imshow(LTPW.get(), interpolation='sinc', cmap='viridis')
    artist2.append([art2])
"""

def show():
    plt.tight_layout()
    plt.show()

""" DISABLED
def update_neural_web_for_fit(W, ax, G, artist):
    art5_1, art5_2, art5_3 = draw_neural_web(W=W, ax=ax, G=G, return_objs=True)
    art5_list = [art5_1] + [art5_2] + list(art5_3.values())
    artist.append(art5_list)
"""

""" DISABLED
def update_decision_boundary_for_fit(ax, x_val, y_val, activations, LTPW, artist1):
    art1_1, art1_2 = plot_decision_boundary(x_val, y_val, activations, LTPW, artist=artist1, ax=ax)
    artist1.append([*art1_1.collections, art1_2])
"""

""" DISABLED
def update_validation_history_for_fit(ax, val_list, artist3):
    val_list_cpu = []
    for i in range(len(val_list)):
        val_list_cpu.append(val_list[i].get())
    period = list(range(1, len(val_list_cpu) + 1))
    art3 = ax.plot(
        period, 
        val_list_cpu, 
        linestyle='--', 
        color='g', 
        marker='o', 
        markersize=6, 
        linewidth=2, 
        label='Validation Accuracy'
    )
    ax.set_title('Validation History')
    ax.set_xlabel('Time')
    ax.set_ylabel('Validation Accuracy')
    ax.set_ylim([0, 1])
    artist3.append(art3)
"""
""" DISABLED
def display_visualization_for_fit(fig, artist_list, interval):
    ani = ArtistAnimation(fig, artist_list, interval=interval, blit=True)
    return ani
"""
def update_neuron_history_for_learner(LTPW, ax1, row, col, class_count, artist5, data, fig1, acc=False, loss=False):

    for j in range(len(class_count)):
        
            if acc != False and loss != False:
                suptitle_info = data + ' Accuracy:' + str(acc) + '\n' + data + ' Loss:' + str(loss) + '\nNeurons Memory:'
            else:
                suptitle_info = 'Neurons Memory:'

            mat = LTPW[j,:].reshape(row, col)

            title_info = f'{j+1}. Neuron'
            
            art5 = ax1[j].imshow(mat.get(), interpolation='sinc', cmap='viridis')

            ax1[j].set_aspect('equal')
            ax1[j].set_xticks([])
            ax1[j].set_yticks([])
            ax1[j].set_title(title_info)

           
            artist5.append([art5])

    fig1.suptitle(suptitle_info, fontsize=16)

    return artist5

def initialize_visualization_for_learner(show_history, neurons_history, neural_web_history, x_train, y_train):

    from .data_ops import find_closest_factors
    viz_objects = {}
    
    if show_history:
        fig, ax = plt.subplots(3, 1, figsize=(6, 8))
        fig.suptitle('Learner History')
        viz_objects['history'] = {
            'fig': fig,
            'ax': ax,
            'artist1': [],
            'artist2': [],
            'artist3': []
        }
    
    if neurons_history:
        row, col = find_closest_factors(len(x_train[0]))
        if row != 0:
            fig1, ax1 = plt.subplots(1, len(y_train[0]), figsize=(18, 14))
        else:
            fig1, ax1 = plt.subplots(1, 1, figsize=(18, 14))
        viz_objects['neurons'] = {
            'fig': fig1,
            'ax': ax1,
            'artists': [],
            'row': row,
            'col': col
        }
    
    if neural_web_history:
        G = nx.Graph()
        fig2, ax2 = plt.subplots(figsize=(18, 4))
        viz_objects['web'] = {
            'fig': fig2,
            'ax': ax2,
            'G': G,
            'artists': []
        }
    
    return viz_objects

def update_history_plots_for_learner(viz_objects, depth_list, loss_list, best_acc_per_depth_list, x_train, final_activations):

    if 'history' not in viz_objects:
        return
    
    hist = viz_objects['history']
    for i in range(len(loss_list)):
        loss_list[i] = loss_list[i].get()

    # Loss plot
    art1 = hist['ax'][0].plot(depth_list, loss_list, color='r', markersize=6, linewidth=2)
    hist['ax'][0].set_title('Train Loss Over Gen')
    hist['artist1'].append(art1)
    
    # Accuracy plot

    for i in range(len(best_acc_per_depth_list)):
        best_acc_per_depth_list[i] = best_acc_per_depth_list[i].get()

    art2 = hist['ax'][1].plot(depth_list, best_acc_per_depth_list, color='g', markersize=6, linewidth=2)
    hist['ax'][1].set_title('Train Accuracy Over Gen')
    hist['artist2'].append(art2)
    
    # Activation shape plot
    x = cp.linspace(cp.min(x_train), cp.max(x_train), len(x_train))
    translated_x_train = cp.copy(x)
    for activation in final_activations:
        translated_x_train += draw_activations(x, activation)
    
    art3 = hist['ax'][2].plot(x.get(), translated_x_train.get(), color='b', markersize=6, linewidth=2)
    hist['ax'][2].set_title('Activation Shape Over Gen')
    hist['artist3'].append(art3)

def display_visualizations_for_learner(viz_objects, best_weights, data, best_acc, test_loss, y_train, interval):

    if 'history' in viz_objects:
        hist = viz_objects['history']
        for _ in range(30):
            hist['artist1'].append(hist['artist1'][-1])
            hist['artist2'].append(hist['artist2'][-1])
            hist['artist3'].append(hist['artist3'][-1])
        
        ani1 = ArtistAnimation(hist['fig'], hist['artist1'], interval=interval, blit=True)
        ani2 = ArtistAnimation(hist['fig'], hist['artist2'], interval=interval, blit=True)
        ani3 = ArtistAnimation(hist['fig'], hist['artist3'], interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
    
    if 'neurons' in viz_objects:
        neurons = viz_objects['neurons']
        for _ in range(10):
            neurons['artists'] = update_neuron_history_for_learner(
                cp.copy(best_weights),
                neurons['ax'],
                neurons['row'],
                neurons['col'],
                y_train[0],
                neurons['artists'],
                data=data,
                fig1=neurons['fig'],
                acc=best_acc,
                loss=test_loss
            )
        
        ani4 = ArtistAnimation(neurons['fig'], neurons['artists'], interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
    
    if 'web' in viz_objects:
        web = viz_objects['web']
        for _ in range(30):
            art5_1, art5_2, art5_3 = draw_neural_web(
                W=best_weights,
                ax=web['ax'],
                G=web['G'],
                return_objs=True
            )
            art5_list = [art5_1] + [art5_2] + list(art5_3.values())
            web['artists'].append(art5_list)
        
        ani5 = ArtistAnimation(web['fig'], web['artists'], interval=interval, blit=True)
        plt.tight_layout()
        plt.show()