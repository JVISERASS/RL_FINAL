"""
Agente DQN para el entorno de almacén (S2).
Un único agente que se entrena progresivamente en los 3 entornos.
Implementado con PyTorch desde cero.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import pickle

from almacen_alu_v1 import WarehouseEnv


# ============================================================================
# RED NEURONAL DQN
# ============================================================================

class DQN(nn.Module):
    """Red neuronal para aproximar Q(s, a)."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Buffer para almacenar experiencias y muestrear mini-batches."""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# INGENIERÍA DE VARIABLES (FEATURE ENGINEERING)
# ============================================================================

def extract_features(obs, just_pick=True):
    """
    Transforma la observación cruda en features útiles para el aprendizaje.
    
    Observación original (11 elementos):
    - obs[0:2]: posición agente (x, y)
    - obs[2:4]: posición objeto 1
    - obs[4:6]: posición objeto 2  
    - obs[6:8]: posición objeto 3
    - obs[8]: agent_has_object
    - obs[9]: collision
    - obs[10]: delivery
    
    Features extraídas:
    - Posición normalizada del agente (2)
    - Distancia y dirección al objeto más cercano (3)
    - Distancia y dirección al área de entrega (3)
    - Flag has_object (1)
    - Distancias a cada objeto (3)
    Total: 12 features
    """
    agent_x, agent_y = obs[0], obs[1]
    has_object = obs[8]
    
    # Posiciones de objetos
    objects = [
        (obs[2], obs[3]),
        (obs[4], obs[5]),
        (obs[6], obs[7])
    ]
    
    # Centro del área de entrega
    delivery_x, delivery_y = 5.0, 9.5
    
    # Normalizar posición del agente [0, 10] -> [-1, 1]
    norm_agent_x = (agent_x / 5.0) - 1.0
    norm_agent_y = (agent_y / 5.0) - 1.0
    
    # Calcular distancias y direcciones a cada objeto
    obj_distances = []
    obj_directions = []
    
    for obj_x, obj_y in objects:
        dx = obj_x - agent_x
        dy = obj_y - agent_y
        dist = np.sqrt(dx**2 + dy**2)
        obj_distances.append(dist)
        obj_directions.append((dx, dy))
    
    # Objeto más cercano
    closest_idx = np.argmin(obj_distances)
    closest_dist = obj_distances[closest_idx] / 14.14  # Normalizar por diagonal
    closest_dx = obj_directions[closest_idx][0] / 10.0
    closest_dy = obj_directions[closest_idx][1] / 10.0
    
    # Distancia y dirección al área de entrega
    dx_del = delivery_x - agent_x
    dy_del = delivery_y - agent_y
    dist_del = np.sqrt(dx_del**2 + dy_del**2) / 14.14
    dir_del_x = dx_del / 10.0
    dir_del_y = dy_del / 10.0
    
    # Distancias normalizadas a cada objeto
    norm_dists = [d / 14.14 for d in obj_distances]
    
    features = np.array([
        norm_agent_x, norm_agent_y,           # Posición agente (2)
        closest_dist, closest_dx, closest_dy,  # Objeto más cercano (3)
        dist_del, dir_del_x, dir_del_y,        # Área de entrega (3)
        has_object,                            # Flag objeto (1)
        norm_dists[0], norm_dists[1], norm_dists[2]  # Distancias objetos (3)
    ], dtype=np.float32)
    
    return features


# ============================================================================
# FUNCIÓN DE RECOMPENSA
# ============================================================================

def compute_reward(obs, prev_obs, action, terminated, just_pick):
    """
    Diseño de recompensa para guiar al agente.
    
    Entorno 1 (just_pick=True): Recoger un objeto
    Entornos 2 y 3 (just_pick=False): Recoger y entregar
    """
    agent_x, agent_y = obs[0], obs[1]
    has_object = obs[8]
    collision = obs[9]
    delivery = obs[10]
    
    prev_has_object = prev_obs[8] if prev_obs is not None else 0
    
    # Calcular distancia al objeto más cercano
    objects = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7])]
    obj_distances = [np.sqrt((ox - agent_x)**2 + (oy - agent_y)**2) for ox, oy in objects]
    min_obj_dist = min(obj_distances)
    
    # Distancia al área de entrega
    delivery_dist = np.sqrt((5.0 - agent_x)**2 + (9.5 - agent_y)**2)
    
    # Calcular distancias previas
    if prev_obs is not None:
        prev_agent_x, prev_agent_y = prev_obs[0], prev_obs[1]
        prev_objects = [(prev_obs[2], prev_obs[3]), (prev_obs[4], prev_obs[5]), (prev_obs[6], prev_obs[7])]
        prev_obj_distances = [np.sqrt((ox - prev_agent_x)**2 + (oy - prev_agent_y)**2) for ox, oy in prev_objects]
        prev_min_obj_dist = min(prev_obj_distances)
        prev_delivery_dist = np.sqrt((5.0 - prev_agent_x)**2 + (9.5 - prev_agent_y)**2)
    else:
        prev_min_obj_dist = min_obj_dist
        prev_delivery_dist = delivery_dist
    
    reward = -1.0  # Penalización por paso
    
    # Penalización por colisión
    if collision > 0.5:
        return -100.0
    
    if just_pick:
        # ENTORNO 1: Solo recoger
        if has_object > 0.5:
            # Éxito: recogió objeto
            return 100.0
        else:
            # Reward shaping: premiar acercarse al objeto
            approach_reward = (prev_min_obj_dist - min_obj_dist) * 5.0
            reward += approach_reward
    else:
        # ENTORNOS 2 y 3: Recoger y entregar
        if delivery > 0.5:
            # Éxito: entregó objeto
            return 200.0
        
        if terminated and has_object < 0.5 and prev_has_object > 0.5:
            # Fracaso: soltó objeto fuera del área
            return -100.0
        
        if has_object > 0.5 and prev_has_object < 0.5:
            # Acaba de coger objeto
            reward += 30.0
        
        if has_object > 0.5:
            # Tiene objeto: premiar acercarse al área de entrega
            approach_reward = (prev_delivery_dist - delivery_dist) * 5.0
            reward += approach_reward
        else:
            # No tiene objeto: premiar acercarse al objeto más cercano
            approach_reward = (prev_min_obj_dist - min_obj_dist) * 5.0
            reward += approach_reward
    
    return reward


def is_success(obs, just_pick):
    """Determina si se ha alcanzado el objetivo según el entorno (sin colisión)."""
    collision = obs[9] > 0.5
    if collision:
        return False
    if just_pick:
        return obs[8] > 0.5  # agent_has_object
    return obs[10] > 0.5     # delivery


# ============================================================================
# AGENTE DQN
# ============================================================================

class DQNAgent:
    """Agente DQN con experience replay y target network."""
    
    def __init__(
        self,
        state_dim=12,
        action_dim=6,  # Número máximo de acciones (se recorta según entorno)
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update=100
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.max_actions = action_dim
        self.action_dim = action_dim
        
        # Redes
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Contador para actualización de target network
        self.steps = 0
        
        # Métricas
        self.losses = []

    def set_action_dim(self, action_dim):
        """Activa únicamente las acciones disponibles en el entorno actual."""
        if action_dim < 1 or action_dim > self.max_actions:
            raise ValueError(f"action_dim debe estar en [1, {self.max_actions}]")
        self.action_dim = action_dim
    
    def select_action(self, state, training=True):
        """Selecciona acción usando política epsilon-greedy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)[:, :self.action_dim]
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Almacena transición en el buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        """Actualiza la red con un mini-batch del buffer."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Muestrear mini-batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q(s, a) actual
        q_values = self.policy_net(states)[:, :self.action_dim]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q(s', a') máximo según target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)[:, :self.action_dim].max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.steps += 1
        
        # Actualizar target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Reduce epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Guarda el modelo."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'action_dim': self.action_dim
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path):
        """Carga el modelo."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.set_action_dim(checkpoint.get('action_dim', self.action_dim))
        print(f"Modelo cargado desde: {path}")


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train(agent, entorno_num, n_episodes=1000, max_steps=500, save_dir="modelos_s2"):
    """
    Entrena el agente en el entorno especificado.
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO EN ENTORNO {entorno_num}")
    print(f"{'='*60}")
    
    # Configurar entorno
    if entorno_num == 1:
        just_pick = True
        random_objects = False
    elif entorno_num == 2:
        just_pick = False
        random_objects = False
    else:  # Entorno 3
        just_pick = False
        random_objects = True

    env = WarehouseEnv(just_pick=just_pick, random_objects=random_objects)
    agent.set_action_dim(env.action_space.n)

    print(f"Configuración: just_pick={just_pick}, random_objects={random_objects}")
    print(f"Acciones disponibles: {agent.action_dim}")
    
    # Métricas
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = extract_features(obs, just_pick)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Seleccionar acción
            action = agent.select_action(state, training=True)
            
            # Ejecutar acción
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_state = extract_features(next_obs, just_pick)
            
            # Calcular recompensa personalizada
            reward = compute_reward(next_obs, obs, action, terminated, just_pick)
            
            done = terminated or truncated
            
            # Almacenar transición
            agent.store_transition(state, action, reward, next_state, float(done))
            
            # Actualizar red
            agent.update()
            
            episode_reward += reward
            obs = next_obs
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Registrar métricas
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Determinar éxito
        success = is_success(obs, just_pick)
        successes.append(success)
        
        # Imprimir progreso
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_success = sum(successes[-100:]) / 100 * 100
            avg_reward = np.mean(recent_rewards)
            print(f"Ep {episode+1}/{n_episodes} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Éxito: {recent_success:.0f}% | "
                  f"ε: {agent.epsilon:.3f}")
    
    env.close()
    
    # Guardar modelo y métricas
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, f"dqn_entorno{entorno_num}.pth"))
    
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'successes': successes
    }
    
    with open(os.path.join(save_dir, f"metrics_entorno{entorno_num}.pkl"), 'wb') as f:
        pickle.dump(metrics, f)
    
    # Graficar
    plot_training(metrics, entorno_num, save_dir)
    
    return metrics


def plot_training(metrics, entorno_num, save_dir):
    """Genera gráficas del entrenamiento."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recompensa
    ax = axes[0]
    rewards = metrics['episode_rewards']
    ax.plot(rewards, alpha=0.3, color='blue')
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa')
    ax.set_title(f'Entorno {entorno_num}: Recompensa')
    ax.grid(True, alpha=0.3)
    
    # Duración
    ax = axes[1]
    lengths = metrics['episode_lengths']
    ax.plot(lengths, alpha=0.3, color='green')
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(lengths)), moving_avg, 'r-', linewidth=2)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Pasos')
    ax.set_title(f'Entorno {entorno_num}: Duración')
    ax.grid(True, alpha=0.3)
    
    # Tasa de éxito
    ax = axes[2]
    successes = metrics['successes']
    success_rate = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        rate = sum(successes[start:i+1]) / (i - start + 1) * 100
        success_rate.append(rate)
    ax.plot(success_rate, color='purple')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Tasa de Éxito (%)')
    ax.set_title(f'Entorno {entorno_num}: Tasa de Éxito')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_entorno{entorno_num}.png"), dpi=150)
    plt.close()
    print(f"Gráficas guardadas en: {save_dir}/training_entorno{entorno_num}.png")


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate(agent, entorno_num, n_episodes=100):
    """Evalúa el agente entrenado."""
    print(f"\n{'='*40}")
    print(f"EVALUACIÓN - ENTORNO {entorno_num}")
    print(f"{'='*40}")
    
    if entorno_num == 1:
        just_pick = True
        random_objects = False
    elif entorno_num == 2:
        just_pick = False
        random_objects = False
    else:
        just_pick = False
        random_objects = True

    env = WarehouseEnv(just_pick=just_pick, random_objects=random_objects)
    agent.set_action_dim(env.action_space.n)
    
    successes = 0
    collisions = 0
    total_rewards = []
    total_steps = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = extract_features(obs, just_pick)
        
        episode_reward = 0
        steps = 0
        
        for step in range(500):
            action = agent.select_action(state, training=False)
            next_obs, _, terminated, truncated, _ = env.step(action)
            
            reward = compute_reward(next_obs, obs, action, terminated, just_pick)
            episode_reward += reward
            steps += 1
            
            obs = next_obs
            state = extract_features(obs, just_pick)
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if is_success(obs, just_pick):
            successes += 1
        if obs[9] > 0.5:  # Colisión
            collisions += 1
    
    env.close()
    
    success_rate = (successes / n_episodes) * 100
    collision_rate = (collisions / n_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print(f"\nResultados ({n_episodes} episodios):")
    print(f"  Tasa de éxito: {success_rate:.1f}%")
    print(f"  Tasa de colisión: {collision_rate:.1f}%")
    print(f"  Recompensa media: {avg_reward:.2f}")
    print(f"  Pasos promedio: {avg_steps:.1f}")
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN para entorno almacén')
    parser.add_argument('--entorno', type=int, default=1, choices=[1, 2, 3],
                        help='Número del entorno (1, 2 o 3)')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Número de episodios de entrenamiento')
    parser.add_argument('--evaluar', action='store_true',
                        help='Solo evaluar modelo existente')
    parser.add_argument('--continuar', action='store_true',
                        help='Continuar entrenamiento desde modelo guardado')
    
    args = parser.parse_args()
    
    # Crear agente (mismo para todos los entornos)
    agent = DQNAgent(
        state_dim=12,
        action_dim=6,  # Máximo de acciones
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.997,
        buffer_size=50000,
        batch_size=64,
        target_update=100
    )
    
    if args.evaluar:
        # Cargar y evaluar
        agent.load(f"modelos_s2/dqn_entorno{args.entorno}.pth")
        evaluate(agent, args.entorno)
    else:
        if args.continuar and os.path.exists(f"modelos_s2/dqn_entorno{args.entorno}.pth"):
            agent.load(f"modelos_s2/dqn_entorno{args.entorno}.pth")
        
        # Entrenar
        train(agent, args.entorno, n_episodes=args.episodes)
        
        # Evaluar
        evaluate(agent, args.entorno)
