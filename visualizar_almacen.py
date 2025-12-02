"""
Visualización con Pygame del agente DQN en el entorno de almacén.
Permite ver el comportamiento del agente en los 3 entornos.
"""

import pygame
import numpy as np
import torch
import sys
import os

from almacen_alu_v1 import WarehouseEnv
from agente_almacen import DQNAgent, extract_features


# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
GREEN = (144, 238, 144)
DARK_GREEN = (0, 128, 0)
BLUE = (0, 100, 255)
BROWN = (139, 69, 19)


class PygameVisualizer:
    """Visualizador del entorno de almacén con Pygame."""
    
    def __init__(self, entorno_num, width=800, height=800):
        pygame.init()
        pygame.display.set_caption(f"Entorno Almacén {entorno_num} - DQN Agent")
        
        self.entorno_num = entorno_num
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height + 80))  # +80 para info
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Escala para mapear coordenadas del entorno (10x10) a la pantalla
        self.scale = width / 10.0
        
        # Configurar entorno
        if entorno_num == 1:
            self.just_pick = True
            self.random_objects = False
        elif entorno_num == 2:
            self.just_pick = False
            self.random_objects = False
        else:
            self.just_pick = False
            self.random_objects = True
        
        # Crear entorno
        self.env = WarehouseEnv(
            just_pick=self.just_pick,
            random_objects=self.random_objects
        )
        
        # Cargar agente
        self.agent = DQNAgent(state_dim=12, action_dim=6)
        model_path = f"modelos_s2/dqn_entorno{entorno_num}.pth"
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Modelo cargado desde: {model_path}")
        else:
            print(f"ERROR: No se encontró el modelo en {model_path}")
            sys.exit(1)
    
    def world_to_screen(self, x, y):
        """Convierte coordenadas del mundo a coordenadas de pantalla."""
        screen_x = int(x * self.scale)
        screen_y = int((10.0 - y) * self.scale)  # Invertir Y
        return screen_x, screen_y
    
    def draw_environment(self, obs, episode, step, reward, total_reward, has_object):
        """Dibuja el entorno completo."""
        # Fondo
        self.screen.fill(WHITE)
        
        # Dibujar área de entrega (verde)
        delivery_rect = pygame.Rect(
            self.world_to_screen(2.5, 11)[0],
            self.world_to_screen(2.5, 11)[1],
            int(5.0 * self.scale),
            int(2.0 * self.scale)
        )
        pygame.draw.rect(self.screen, GREEN, delivery_rect)
        pygame.draw.rect(self.screen, DARK_GREEN, delivery_rect, 2)
        
        # Dibujar estanterías (marrón)
        shelves = [(1.9, 1.0, 0.2, 5.0), (4.9, 1.0, 0.2, 5.0), (7.9, 1.0, 0.2, 5.0)]
        for sx, sy, sw, sh in shelves:
            shelf_rect = pygame.Rect(
                self.world_to_screen(sx, sy + sh)[0],
                self.world_to_screen(sx, sy + sh)[1],
                int(sw * self.scale),
                int(sh * self.scale)
            )
            pygame.draw.rect(self.screen, BROWN, shelf_rect)
            pygame.draw.rect(self.screen, BLACK, shelf_rect, 2)
        
        # Dibujar objetos (azul)
        objects = [
            (obs[2], obs[3]),
            (obs[4], obs[5]),
            (obs[6], obs[7])
        ]
        for ox, oy in objects:
            # Solo dibujar si no está en la posición del agente (no recogido)
            agent_x, agent_y = obs[0], obs[1]
            if not (abs(ox - agent_x) < 0.1 and abs(oy - agent_y) < 0.1):
                screen_pos = self.world_to_screen(ox, oy)
                pygame.draw.circle(self.screen, BLUE, screen_pos, int(0.2 * self.scale))
        
        # Dibujar agente
        agent_color = RED if has_object else ORANGE
        agent_screen = self.world_to_screen(obs[0], obs[1])
        pygame.draw.circle(self.screen, agent_color, agent_screen, int(0.25 * self.scale))
        pygame.draw.circle(self.screen, BLACK, agent_screen, int(0.25 * self.scale), 2)
        
        # Dibujar borde del entorno
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width, self.height), 3)
        
        # Panel de información
        info_y = self.height + 10
        
        # Título
        title = f"Entorno {self.entorno_num}"
        if self.entorno_num == 1:
            title += " - Solo Recoger"
        elif self.entorno_num == 2:
            title += " - Recoger y Entregar (Fijo)"
        else:
            title += " - Recoger y Entregar (Aleatorio)"
        title_surf = self.font_large.render(title, True, BLACK)
        self.screen.blit(title_surf, (10, info_y))
        
        # Estadísticas
        stats = f"Episodio: {episode}  |  Paso: {step}  |  Recompensa: {total_reward:.1f}  |  Objeto: {'Sí' if has_object else 'No'}"
        stats_surf = self.font.render(stats, True, BLACK)
        self.screen.blit(stats_surf, (10, info_y + 35))
        
        # Instrucciones
        instr = "ESC: Salir  |  ESPACIO: Pausar  |  R: Reiniciar episodio"
        instr_surf = self.font.render(instr, True, GRAY)
        self.screen.blit(instr_surf, (self.width - 350, info_y + 35))
        
        pygame.display.flip()
    
    def run(self, n_episodes=5, delay=100):
        """Ejecuta la visualización."""
        print(f"\nVisualizando {n_episodes} episodios del Entorno {self.entorno_num}")
        print("Controles: ESC=Salir, ESPACIO=Pausar, R=Reiniciar")
        
        successes = 0
        total_steps = []
        total_rewards = []
        
        running = True
        paused = False
        
        for episode in range(n_episodes):
            if not running:
                break
            
            obs, _ = self.env.reset()
            state = extract_features(obs, self.just_pick)
            
            episode_reward = 0
            step = 0
            done = False
            has_object = False
            
            while not done and running:
                # Manejar eventos
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            done = True  # Reiniciar episodio
                
                if paused:
                    self.clock.tick(10)
                    continue
                
                # Seleccionar acción
                action = self.agent.select_action(state, training=False)
                
                # Ejecutar acción
                next_obs, _, terminated, truncated, _ = self.env.step(action)
                has_object = next_obs[8] > 0.5
                
                # Calcular recompensa para mostrar
                from agente_almacen import compute_reward
                reward = compute_reward(next_obs, obs, action, terminated, self.just_pick)
                episode_reward += reward
                
                # Dibujar
                self.draw_environment(next_obs, episode + 1, step, reward, episode_reward, has_object)
                
                obs = next_obs
                state = extract_features(obs, self.just_pick)
                step += 1
                done = terminated or truncated
                
                self.clock.tick(1000 // delay)
            
            if running and not paused:
                # Determinar resultado
                success = episode_reward > 50
                if success:
                    successes += 1
                    result = "ÉXITO"
                else:
                    result = "FRACASO"
                
                total_steps.append(step)
                total_rewards.append(episode_reward)
                
                print(f"Episodio {episode + 1}: {result} en {step} pasos, Recompensa: {episode_reward:.1f}")
                
                # Pausa breve entre episodios
                pygame.time.wait(500)
        
        # Resumen final
        if total_rewards:
            print(f"\n{'='*50}")
            print("RESUMEN")
            print(f"{'='*50}")
            print(f"Éxitos: {successes}/{len(total_rewards)} ({100*successes/len(total_rewards):.1f}%)")
            print(f"Pasos promedio: {np.mean(total_steps):.1f}")
            print(f"Recompensa promedio: {np.mean(total_rewards):.1f}")
        
        pygame.quit()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualizar agente DQN en almacén')
    parser.add_argument('--entorno', type=int, default=1, choices=[1, 2, 3],
                        help='Número del entorno (1, 2 o 3)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Número de episodios a visualizar')
    parser.add_argument('--delay', type=int, default=50,
                        help='Delay entre frames en ms (menor = más rápido)')
    
    args = parser.parse_args()
    
    visualizer = PygameVisualizer(args.entorno)
    visualizer.run(n_episodes=args.episodes, delay=args.delay)


if __name__ == "__main__":
    main()
