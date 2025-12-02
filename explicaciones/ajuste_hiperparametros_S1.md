# Ajuste de Hiperparámetros - Sesión 1: Navegación con Tile Coding

## 1. Valores Iniciales de la Práctica

Los valores iniciales proporcionados en la práctica estaban **deliberadamente mal configurados** (como indica el PDF):

```python
# Tile Coding
n_tiles_width = 1
n_tiles_height = 1
n_tilings = 1

# Agente SARSA
learning_rate = 1
discount_factor = 0.99
epsilon = 0.5

# Estrategia de exploración
decay_start = 0.9   # Empezar decay muy tarde
decay_rate = 0.9    # Decay muy agresivo
min_epsilon = 0.5   # Epsilon mínimo muy alto

# Recompensa
reward = -50  # Constante para todo
```

### Problemas con estos valores:

| Parámetro | Problema |
|-----------|----------|
| `n_tiles = 1x1` | Sin discretización, todo el espacio es un único tile → no hay generalización útil |
| `n_tilings = 1` | Sin solapamiento, mala resolución del espacio |
| `learning_rate = 1` | Demasiado alto, causa inestabilidad en el aprendizaje |
| `decay_start = 0.9` | El agente explora casi todo el entrenamiento y apenas explota |
| `min_epsilon = 0.5` | Nunca converge a política determinista (50% acciones aleatorias siempre) |
| `reward = -50` | Sin distinción entre éxito, fracaso o paso normal |

---

## 2. Primera Iteración de Ajustes

### 2.1 Tile Coding

**Cambio:** `1x1 tiles, 1 tiling` → `10x10 tiles, 8 tilings`

**Razonamiento:**
- El entorno es de 10x10 metros
- Con 10 tiles por dimensión, cada tile cubre 1m² → buena granularidad
- 8 tilings con offset (3,1) proporcionan:
  - Mejor generalización (experiencia en un punto afecta áreas cercanas)
  - Resolución efectiva de 10×8 = 80 posiciones distinguibles por eje
  - Suavizado del aprendizaje (menos saltos bruscos en Q-values)

### 2.2 Learning Rate

**Cambio:** `1.0` → `0.1/8 ≈ 0.0125`

**Razonamiento:**
- En tile coding, el learning rate debe dividirse entre el número de tilings
- Fórmula estándar: `α_efectivo = α / n_tilings`
- Esto evita que las actualizaciones sean demasiado grandes
- Con 8 tilings activos simultáneamente, cada uno recibe `α/8` de la actualización

### 2.3 Función de Recompensa

**Cambio:** `reward = -50` (constante) → Sistema diferenciado

```python
if collision:
    reward = -100  # Penalización por colisión
elif target:
    reward = 100   # Recompensa por llegar al objetivo
else:
    reward = -1    # Pequeña penalización por paso
```

**Razonamiento:**
- **Colisión (-100):** Señal clara de fracaso, el agente aprende a evitar paredes
- **Objetivo (+100):** Señal clara de éxito, incentiva llegar rápido
- **Paso (-1):** Penalización pequeña que incentiva eficiencia (menos pasos = mejor)

### 2.4 Estrategia de Exploración

**Cambio:**
```python
# Antes
decay_start = 0.9
decay_rate = 0.9
min_epsilon = 0.5

# Después
decay_start = 0.5
decay_rate = 0.995
min_epsilon = 0.01
```

**Razonamiento:**
- `decay_start = 0.5`: Empezar a reducir epsilon a mitad del entrenamiento
  - Primera mitad: exploración intensa para descubrir el espacio
  - Segunda mitad: explotación gradual de lo aprendido
- `decay_rate = 0.995`: Reducción suave y gradual
  - Decay agresivo (0.9) causaría caída brusca
  - 0.995 permite ~1000 episodios para transición suave
- `min_epsilon = 0.01`: Casi determinista al final
  - Necesario para que la política evaluada sea consistente
  - 1% de exploración residual evita quedarse atascado

---

## 3. Segunda Iteración (Optimización Final)

Tras ver los primeros resultados (77% success rate), identifiqué áreas de mejora:

### 3.1 Tile Coding Refinado

**Cambio:** `10x10 tiles, 8 tilings` → `8x8 tiles, 16 tilings`

**Razonamiento:**
- **Menos tiles (8x8):** Cada tile cubre más área → mejor generalización
- **Más tilings (16):** Compensa la menor resolución con más solapamiento
- **Trade-off:** 
  - Más tiles = más precisión pero menos generalización
  - Más tilings = mejor resolución efectiva pero más memoria
- Con 16 tilings: resolución efectiva de 8×16 = 128 posiciones por eje

### 3.2 Learning Rate Ajustado

**Cambio:** `0.1/8` → `0.2/16 = 0.0125`

**Razonamiento:**
- Mismo learning rate efectivo (~0.0125)
- Pero distribuido entre más tilings
- Permite actualizaciones más suaves con mejor cobertura

### 3.3 Recompensas Amplificadas

**Cambio:** `±100` → `±200`

```python
if collision:
    reward = -200  # Penalización más fuerte
elif target:
    reward = 200   # Recompensa más alta
else:
    reward = -1    # Igual
```

**Razonamiento:**
- Mayor contraste entre éxito/fracaso
- Señal más clara para el agente
- Acelera la convergencia al hacer las consecuencias más evidentes

### 3.4 Epsilon Más Explorador Inicialmente

**Cambio:** `epsilon = 0.3` → `epsilon = 0.4`

**Razonamiento:**
- Más exploración inicial = mejor cobertura del espacio de estados
- El agente descubre más trayectorias posibles antes de explotar

### 3.5 Decay Más Suave

**Cambio:**
```python
decay_start = 0.5 → 0.3    # Empezar antes
decay_rate = 0.995 → 0.9995 # Más gradual
min_epsilon = 0.01 → 0.005  # Más determinista
```

**Razonamiento:**
- `decay_start = 0.3`: Con más exploración inicial (ε=0.4), podemos empezar decay antes
- `decay_rate = 0.9995`: Transición muy suave de exploración a explotación
- `min_epsilon = 0.005`: Política casi completamente determinista para evaluación

### 3.6 Max Steps Reducido

**Cambio:** `max_steps = 20000` → `max_steps = 500`

**Razonamiento:**
- 20000 pasos es excesivo para un entorno de 10x10m
- Con velocidad de 0.25m/paso, cruzar en diagonal ≈ 57 pasos
- 500 pasos es suficiente para cualquier trayectoria razonable
- Evita que episodios fallidos duren demasiado (desperdicio de tiempo)

### 3.7 Más Episodios de Entrenamiento

**Cambio:** `10000` → `15000` episodios

**Razonamiento:**
- Más tiempo para que el decay de epsilon surta efecto
- Asegura convergencia completa antes de evaluación

---

## 4. Resultados Comparativos

| Configuración | Success Rate | Avg Return | Convergencia |
|---------------|--------------|------------|--------------|
| **Inicial (práctica)** | ~0% | Negativo | No converge |
| **Primera iteración** | ~77% | 77 | ~9000 eps |
| **Optimización final** | **100%** | **179.68** | ~11000 eps |

---

## 5. Lecciones Aprendidas

### 5.1 Tile Coding
- El balance entre número de tiles y tilings es crucial
- Más tilings siempre ayuda (hasta límite de memoria)
- El offset (3,1) evita artefactos de alineación

### 5.2 Learning Rate
- **Regla de oro:** `α_efectivo = α_base / n_tilings`
- Valores típicos: 0.1-0.5 para `α_base`

### 5.3 Exploración
- La estrategia de decay es más importante que el epsilon inicial
- Decay gradual > decay agresivo
- Siempre terminar con epsilon muy bajo para evaluación

### 5.4 Recompensas
- Contraste claro entre éxito y fracaso
- Penalización por paso pequeña pero presente
- Recompensas sparse (solo al final) funcionan bien con tile coding

### 5.5 Episodios
- Más episodios casi siempre ayuda
- Pero hay punto de rendimientos decrecientes
- Monitorizar success rate para detectar convergencia

---

## 6. Configuración Final Recomendada

```python
# Tile Coding
n_tiles_width = 8
n_tiles_height = 8
n_tilings = 16

# Agente SARSA
learning_rate = 0.2 / n_tilings  # = 0.0125
discount_factor = 0.99
epsilon = 0.4

# Estrategia de exploración
decay_start = 0.3
decay_rate = 0.9995
min_epsilon = 0.005

# Entorno
max_steps = 500

# Recompensas
reward_collision = -200
reward_target = 200
reward_step = -1

# Entrenamiento
num_episodes = 15000
```

Esta configuración logra **100% success rate** con un retorno promedio de **~180** (máximo teórico: 200).
