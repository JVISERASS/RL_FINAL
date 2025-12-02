"""
Visualización del agente entrenado con Pygame
"""
import pygame
import numpy as np
import pickle
import sys
import time

# Importar clases necesarias para deserializar el agente
from agente import SarsaAgent
from entorno_navegacion import Navegacion
from representacion import FeedbackConstruction

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
LIGHT_GREEN = (144, 238, 144)
GRAY = (128, 128, 128)

class PygameVisualizer:
    def __init__(self, env, scale=60):
        """
        Visualizador del entorno con Pygame
        
        Args:
            env: Entorno de navegación
            scale: Píxeles por unidad del entorno (1 metro = scale píxeles)
        """
        self.env = env
        self.scale = scale
        self.width = int(env.width * scale)
        self.height = int(env.height * scale)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Navegación - Agente SARSA")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
    def _to_screen(self, pos):
        """Convierte coordenadas del entorno a coordenadas de pantalla"""
        return (int(pos[0] * self.scale), int(self.height - pos[1] * self.scale))
    
    def _scale_size(self, size):
        """Escala un tamaño del entorno a píxeles"""
        return (int(size[0] * self.scale), int(size[1] * self.scale))
    
    def render(self, episode=0, step=0, total_reward=0):
        """Renderiza el estado actual del entorno"""
        # Manejar eventos de pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # Limpiar pantalla
        self.screen.fill(WHITE)
        
        # Dibujar obstáculos
        for obstacle in self.env.obstacles:
            x, y, w, h = obstacle
            rect_pos = self._to_screen((x, y + h))
            rect_size = self._scale_size((w, h))
            pygame.draw.rect(self.screen, BROWN, (*rect_pos, *rect_size))
            pygame.draw.rect(self.screen, BLACK, (*rect_pos, *rect_size), 2)
        
        # Dibujar área objetivo
        tx, ty, tw, th = self.env.target_area
        target_pos = self._to_screen((tx, ty + th))
        target_size = self._scale_size((tw, th))
        pygame.draw.rect(self.screen, LIGHT_GREEN, (*target_pos, *target_size))
        pygame.draw.rect(self.screen, GREEN, (*target_pos, *target_size), 3)
        
        # Dibujar agente
        agent_screen_pos = self._to_screen(self.env.agent_pos)
        agent_radius = int(self.env.agent_radius * self.scale)
        pygame.draw.circle(self.screen, ORANGE, agent_screen_pos, agent_radius)
        pygame.draw.circle(self.screen, BLACK, agent_screen_pos, agent_radius, 2)
        
        # Dibujar bordes del entorno
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width, self.height), 3)
        
        # Mostrar información
        info_text = f"Episodio: {episode} | Paso: {step} | Recompensa: {total_reward:.0f}"
        text_surface = self.font.render(info_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        # Estado del agente
        if self.env.target:
            status = "¡OBJETIVO ALCANZADO!"
            color = GREEN
        elif self.env.collision:
            status = "¡COLISIÓN!"
            color = RED
        else:
            status = "Navegando..."
            color = GRAY
        
        status_surface = self.font.render(status, True, color)
        self.screen.blit(status_surface, (10, 50))
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def close(self):
        pygame.quit()


def run_visualization(num_episodes=5, delay=0.05):
    """
    Ejecuta la visualización del agente entrenado
    
    Args:
        num_episodes: Número de episodios a visualizar
        delay: Tiempo de espera entre pasos (segundos)
    """
    # Cargar el agente entrenado
    try:
        with open('agente_grupo_xx_a.pkl', 'rb') as f:
            agent = pickle.load(f)
        print("Agente cargado correctamente")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'agente_grupo_xx_a.pkl'")
        print("Primero entrena el agente ejecutando 'python agente.py'")
        return
    
    # Crear visualizador
    vis = PygameVisualizer(agent.env, scale=60)
    
    print(f"\nVisualizando {num_episodes} episodios...")
    print("Presiona ESC o cierra la ventana para salir\n")
    
    results = []
    
    for episode in range(num_episodes):
        state, _ = agent.env.reset()
        total_reward = 0
        step = 0
        terminated = False
        truncated = False
        
        # Pausa inicial para ver posición de inicio
        vis.render(episode + 1, step, total_reward)
        time.sleep(0.5)
        
        while not terminated and not truncated:
            # Obtener acción del agente (política greedy)
            action = agent.get_action(state, epsilon=0)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            total_reward += reward
            step += 1
            
            # Renderizar
            vis.render(episode + 1, step, total_reward)
            time.sleep(delay)
            
            state = next_state
        
        # Resultado del episodio
        if agent.env.target:
            result = "ÉXITO"
        elif agent.env.collision:
            result = "COLISIÓN"
        else:
            result = "TRUNCADO"
        
        results.append((result, step, total_reward))
        print(f"Episodio {episode + 1}: {result} en {step} pasos, Recompensa: {total_reward:.0f}")
        
        # Pausa entre episodios
        time.sleep(1)
    
    # Resumen
    print("\n" + "="*50)
    print("RESUMEN")
    print("="*50)
    successes = sum(1 for r in results if r[0] == "ÉXITO")
    avg_steps = np.mean([r[1] for r in results if r[0] == "ÉXITO"]) if successes > 0 else 0
    avg_reward = np.mean([r[2] for r in results])
    print(f"Éxitos: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Pasos promedio (éxitos): {avg_steps:.1f}")
    print(f"Recompensa promedio: {avg_reward:.1f}")
    
    vis.close()


if __name__ == "__main__":
    # Número de episodios a visualizar
    n_episodes = 10
    
    # Velocidad de visualización (segundos entre pasos)
    # Menor = más rápido
    speed = 0.03
    
    run_visualization(num_episodes=n_episodes, delay=speed)
