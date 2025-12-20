import os
import sys
import subprocess

def run_script(script_name):
    """Ejecuta un script de Python y verifica si terminó bien."""
    print(f"\n{'='*60}")
    print(f"▶️  EJECUTANDO: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Usamos subprocess.run para esperar a que termine antes de seguir
        result = subprocess.run([sys.executable, script_name], check=True)
        if result.returncode == 0:
            print(f"✅ {script_name} completado con éxito.")
            return True
        else:
            print(f"❌ {script_name} terminó con errores.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error crítico ejecutando {script_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def main():
    print("\n" + "#"*60)
    print("   SISTEMA DE TRADING ALGORÍTMICO USD/PEN (GRUPO 3)")
    print("   Enfoque: HMM (No Supervisado) + RL (Q-Learning)")
    print("#"*60 + "\n")
    
    # 1. Preprocesamiento (Carga, Limpieza, PCA, Escalado)
    if not run_script("preprocessing.py"):
        print("⛔ Deteniendo ejecución por error en preprocesamiento.")
        return

    # 2. Modelo HMM (Detección de Regímenes)
    if not run_script("hmm_model.py"):
        print("⛔ Deteniendo ejecución por error en HMM.")
        return

    # 3. Entrenamiento RL (Agente)
    # Aquí es donde ocurre la magia del Reinforcement Learning
    if not run_script("rl_agent.py"):
        print("⛔ Deteniendo ejecución por error en entrenamiento RL.")
        return

    # 4. Evaluación y Gráficos
    if not run_script("evaluation.py"):
        print("⛔ Deteniendo ejecución por error en evaluación.")
        return
    
    print("\n" + "#"*60)
    print("🎉 ¡PROCESO COMPLETO! 🎉")
    print("Revisa la carpeta 'output_images' y el archivo 'evaluation_metrics.csv'.")
    print("#"*60)

if __name__ == "__main__":
    main()