import numpy as np
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

class SarsaAgent:
    """
    SarsaAgent is an implementation of the SARSA(0) algorithm for reinforcement learning.
    Attributes:
        env (gym.Env): The environment in which the agent operates.
        feedback (object): An object that processes observations from the environment.
        learning_rate (float): The learning rate for updating the weights.
        discount_factor (float): The discount factor for future rewards.
        epsilon (float): The probability of choosing a random action (exploration rate).
        num_actions (int): The number of possible actions in the environment.
        feature_size (int): The size of the feature vector for each state.
        weights (list of np.ndarray): The weights for each action.
    Methods:
        __init__(env, gateway, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
            Initializes the SarsaAgent with the given parameters.
        get_action(state, epsilon=None):
            Returns an action based on the epsilon-greedy policy.
        get_q_values(state):
            Computes the Q-values for all actions given the current state.
        update(state, action, reward, next_state, next_action):
            Updates the weights based on the SARSA update rule.
        train(num_episodes):
            Trains the agent for a specified number of episodes.
        evaluate(num_episodes):
            Evaluates the agent's performance over a specified number of episodes.
    """
    def __init__(self, env, feedback, learning_rate=0.5, discount_factor=0.9, epsilon=0.5):
        # Mejor no toques estas líneas
        self.env = env
        self.feedback = feedback
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = env.action_space.n
        self.feature_size = feedback.iht.size # Si vas a añadir más variables (features) además del tile coding, resérvales espacio aquí.
        ##############################

        # Te damos los pesos inicializados a cero. Pero esto es arbitrario. Lo puedes cambiar si quieres.
        self.weights = [np.zeros(self.feature_size) for _ in range(self.num_actions)]
        
        # Tendrás que usar estrategias para monitorizar el aprendizaje del agente.
        # Añade aquí los atributos que necesites para hacerlo.
        self.episode_returns = []  # Historial de retornos por episodio
        self.episode_lengths = []  # Longitud de trayectorias
        self.success_count = 0     # Contador de éxitos
        self.collision_count = 0   # Contador de colisiones
        ##############################

    def get_action(self, state, epsilon=None):
        """
        Selects an action based on the epsilon-greedy policy.
        Parameters:
        state (object): The current state of the environment.
        epsilon (float, optional): The probability of selecting a random action. 
                                   If None, the default epsilon value is used.
        Returns:
        int: The selected action.
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def get_q_values(self, state):
        """
        Computes the Q-values of all actions for a given state.

        Parameters:
        state (object): The current state for which Q-values need to be computed.

        Returns:
        np.ndarray: A numpy array of Q-values for each action in the given state.
        """
        features = self.feedback.process_observation(state)
        q_values = np.zeros(self.num_actions)
        # Calcula los valores de cada acción para el estado dado como argumento (aproximación
        # lineal). Añade aquí tu código
        # Q(s,a) = sum of weights at active tile indices for action a
        for a in range(self.num_actions):
            q_values[a] = np.sum(self.weights[a][features])

        ###################################
        return q_values
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        """
        Update the weights for the given state-action pair using the SARSA(0) algorithm.
        Parameters:
        state (object): The current state.
        action (int): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (object): The state resulting from taking the action.
        next_action (int): The action to be taken in the next state.
        Returns:
        None
        """
        qs_current = self.get_q_values(state)       
        q_current = qs_current[action]
        # td_error
        if terminated:
            td_error = reward - q_current
        else:
            qs_next = self.get_q_values(next_state)
            q_next = qs_next[next_action]            
            td_error = reward + self.discount_factor * q_next - q_current
        # Añade aquí tu código para actualizar los pesos del agente
        # Actualización SARSA(0): w_a = w_a + alpha * td_error * gradient
        # Con tile coding, el gradiente es 1 para tiles activos, 0 para el resto
        features = self.feedback.process_observation(state)
        self.weights[action][features] += self.learning_rate * td_error
        #############################################
        
    def train(self, num_episodes):
        """
        Train the agent using the SARSA(0) algorithm.
        Parameters:
        num_episodes (int): The number of episodes to train the agent.
        The method runs the training loop for the specified number of episodes.
        In each episode, the agent interacts with the environment, selects actions
        based on the current policy, and updates the policy using the SARSA(0) update rule.
        The total reward for each episode is printed every 100 episodes.
        Returns:
        None
        """
        #Juega con estos tres hiperparámetros
        decay_start = 0.3  # Empezar decay más temprano para más tiempo explotando
        decay_rate = 0.9995 # Decay muy gradual para convergencia suave
        min_epsilon = 0.005 # Epsilon muy bajo para política casi determinista
        ####################################
        for episode in range(num_episodes):
            #Set-up del episodio
            state, _ = self.env.reset()
            #Decrecimiento exponencial de epsilon hasta valor mínimo desde comienzo marcado
            if episode >= num_episodes*decay_start:
                self.epsilon *= decay_rate
                self.epsilon = np.max([min_epsilon,self.epsilon])
            #Primera acción
            action = self.get_action(state, self.epsilon)            
            n_steps = 0
            #Generación del episodio
            total_undiscounted_return = 0
            while True:                                        
                next_state, reward, terminated, truncated, _ = self.env.step(action)  
                total_undiscounted_return += reward          
                next_action = self.get_action(next_state, self.epsilon)
                self.update(state, action, reward, next_state, next_action, terminated)    
                state = next_state
                action = next_action                
                n_steps += 1
                if terminated or truncated:
                    break
            
            # Monitorización del aprendizaje
            self.episode_returns.append(total_undiscounted_return)
            self.episode_lengths.append(n_steps)
            if total_undiscounted_return > 0:  # Llegó al objetivo
                self.success_count += 1
            elif total_undiscounted_return <= -100:  # Colisión
                self.collision_count += 1

            #Aquí también puedes cambiar la frecuencia con la que muestras
            #los resultados en la consola, e incluso deshabilitarla.
            episodes_update = 1000
            if episode % episodes_update == 0:
                # Calcular estadísticas de los últimos 1000 episodios
                recent_returns = self.episode_returns[-episodes_update:] if len(self.episode_returns) >= episodes_update else self.episode_returns
                avg_return = np.mean(recent_returns)
                success_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns) * 100
                print(f"Episode {episode}, Avg Return: {avg_return:.1f}, Success Rate: {success_rate:.1f}%, Epsilon: {self.epsilon:.4f}")
                #puedes salvar el estado actual del agente, si te viene bien    

    
    def evaluate(self, num_episodes):
        """
        Evaluate the agent's performance over a specified number of episodes.
        Parameters:
        num_episodes (int): The number of episodes to run the evaluation.
        Returns:
        float: The average reward obtained over the specified number of episodes.
        This method runs the agent in the environment for a given number of episodes
        using a greedy policy (epsilon=0). It collects the total reward for each episode
        and computes the average reward over all episodes. The average reward is printed
        and returned.
        Note:
        - The environment is reset at the beginning of each episode.
        - The agent's action is determined by the `get_action` method with epsilon set to 0.
        """
        total_returns = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_undiscounted_return = 0
            terminated = False
            
            while not terminated:
                action = self.get_action(state, epsilon=0)  # Política completamente greedy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if num_episodes <= 5:  # Solo renderizar si son pocos episodios
                    self.env.render()
                state = next_state
                total_undiscounted_return += reward
                if truncated:
                    break
            
            total_returns.append(total_undiscounted_return)
        
        avg_return = np.mean(total_returns)
        success_rate = sum(1 for r in total_returns if r > 0) / num_episodes * 100
        print(f"Average undiscounted return over {num_episodes} episodes: {avg_return:.2f}")
        print(f"Success rate: {success_rate:.1f}%")
        return avg_return


if __name__ == "__main__":
    #instanciamos entorno, representación y agente
    #No tocar
    env = Navegacion()
    warehouse_width = 10.0
    warehouse_height = 10.0
    ################
    #diseñar los tiles
    n_tiles_width = 8   # Menos tiles pero más tilings = mejor generalización
    n_tiles_height = 8
    n_tilings = 16      # Más tilings para mejor resolución
    
    target_area = (2.5, 8, 1.0, 2.0)

    feedback = FeedbackConstruction((warehouse_width, warehouse_height), 
                                 (n_tiles_width, n_tiles_height), 
                                 n_tilings, target_area)
    
    # Learning rate dividido por n_tilings (estándar en tile coding)
    agent = SarsaAgent(env, feedback, learning_rate=0.2/n_tilings, discount_factor=0.99, epsilon=0.4)
    
    # Train the agent
    agent.train(num_episodes=15000)
    
    #save the agent object into memory    
    with open('agente_grupo_xx_a.pkl', 'wb') as f:
        pickle.dump(agent, f)

    # Evaluate the agent with more episodes
    agent.evaluate(num_episodes=100)

    