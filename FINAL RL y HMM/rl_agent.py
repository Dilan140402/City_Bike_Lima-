import numpy as np
import pandas as pd
import random
import os
from collections import deque
import time
import joblib

# --- CONFIGURACIÓN DE COMPATIBILIDAD ---
# Intentamos importar TensorFlow. Si falla (DLL error), usamos Scikit-Learn.
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    # Desactivar logs molestos de OneDNN
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    BACKEND = 'tensorflow'
    print("✅ TensorFlow detectado. Usando Red Neuronal Profunda (DQN).")
except ImportError:
    from sklearn.neural_network import MLPRegressor
    BACKEND = 'sklearn'
    print("⚠️ TensorFlow no encontrado o error de DLL. Usando Scikit-Learn (MLP) como respaldo.")

from trading_env import TradingEnv

# Semillas para reproducibilidad
np.random.seed(42)
random.seed(42)
if BACKEND == 'tensorflow':
    tf.random.set_seed(42)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hiperparámetros
        self.gamma = 0.95    # Factor de descuento (valor futuro)
        self.epsilon = 1.0   # Tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Construir modelo según el backend disponible
        self.model = self._build_model()
        self.target_model = self._build_model() if BACKEND == 'tensorflow' else None
        self.update_target_model()

    def _build_model(self):
        """Construye la Red Neuronal."""
        if BACKEND == 'tensorflow':
            # Arquitectura optimizada para ser rápida (Cap 18 Géron)
            model = Sequential([
                Input(shape=(self.state_size,)),
                Dense(24, activation='relu'),
                Dense(24, activation='relu'),
                Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
        else:
            # Versión ligera con Scikit-Learn (si falla TF)
            model = MLPRegressor(hidden_layer_sizes=(24, 24), activation='relu', 
                                 solver='adam', learning_rate_init=self.learning_rate, max_iter=1)
            # Inicializar con fit parcial para evitar error de "no fitteado"
            model.partial_fit([np.zeros(self.state_size)], [np.zeros(self.action_size)])
            return model

    def update_target_model(self):
        """Copia los pesos del modelo principal al target (estabilidad)."""
        if BACKEND == 'tensorflow':
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Política Epsilon-Greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if BACKEND == 'tensorflow':
            act_values = self.model.predict(state[np.newaxis, :], verbose=0)
        else:
            act_values = self.model.predict([state])
            
        return np.argmax(act_values[0])

    def replay(self):
        """Entrenamiento con Experience Replay"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Preparar arrays para entrenamiento por lotes (más rápido)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Predicciones
        if BACKEND == 'tensorflow':
            target = self.model.predict(states, verbose=0)
            target_next = self.target_model.predict(next_states, verbose=0)
        else:
            target = self.model.predict(states)
            target_next = self.model.predict(next_states) # Sklearn no usa target net separado simple

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # Entrenar modelo
        if BACKEND == 'tensorflow':
            self.model.fit(states, target, epochs=1, verbose=0)
        else:
            self.model.partial_fit(states, target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, name):
        if BACKEND == 'tensorflow':
            self.model.save(name + ".keras")
        else:
            joblib.dump(self.model, name + ".pkl")

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # Cargar datos
    data_path = "processed_data_with_hmm.csv"
    if not os.path.exists(data_path):
        print("❌ Error: No se encuentra 'processed_data_with_hmm.csv'. Ejecuta hmm_model.py primero.")
        exit()
        
    df = pd.read_csv(data_path)
    
    # Dividir Train/Test (Entrenar hasta 2024, probar en 2025)
    # Nota: Ajusta la fecha según tus datos.
    split_date = '2025-01-01'
    df['Date'] = pd.to_datetime(df['Date']) if 'Date' in df.columns else pd.to_datetime(df.index)
    
    train_df = df[df['Date'] < split_date]
    if len(train_df) == 0:
        print("⚠️ Advertencia: Pocos datos. Usando el 80% inicial para entrenar.")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]

    # Inicializar Entorno y Agente
    env = TradingEnv(train_df)
    state_size = env.observation_space.shape[0]
    action_size = int(env.action_space.n)
    
    agent = DQNAgent(state_size, action_size)
    
    # --- BUCLE DE ENTRENAMIENTO ---
    EPISODES = 10  # Reducido para que termine rápido en tu prueba
    
    print(f"\n🚀 INICIANDO ENTRENAMIENTO ({EPISODES} Episodios) con Backend: {BACKEND}")
    print("="*60)
    
    start_time = time.time()
    
    for e in range(EPISODES):
        # Reiniciar entorno (Gymnasium devuelve state, info)
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            # 1. Acción
            action = agent.act(state)
            
            # 2. Paso en el entorno
            step_result = env.step(action)
            # Manejar diferencias de versiones de Gym
            if len(step_result) == 5:
                next_state, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                next_state, reward, done, info = step_result
            
            # 3. Guardar experiencia
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            # 4. Entrenar (Replay)
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                
            # --- FEEDBACK VISUAL PARA QUE NO PIENSES QUE SE COLGÓ ---
            if step % 100 == 0:
                print(f"   > Episodio {e+1} | Día {step} | Recompensa Acum: {total_reward:.2f}", end='\r')

        # Fin del episodio
        agent.update_target_model()
        print(f"\n✅ Episodio: {e+1}/{EPISODES} | Score Final: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

    print("="*60)
    print(f"🏁 Entrenamiento finalizado en {(time.time() - start_time)/60:.1f} minutos.")
    
    # Guardar modelo
    agent.save("dqn_model_final")
    print("💾 Modelo guardado como 'dqn_model_final'.")