from flask import Flask, request, jsonify
from flask_cors import CORS
import pymysql
from flasgger import Swagger
import os

app = Flask(__name__)
CORS(app)

# --- Swagger config (puedes editar título/version) ---
swagger_template = {
    "info": {
        "title": "CityBike - API1 (Usuarios)",
        "version": "1.0",
        "description": "API 1 para la práctica: CRUD simple de usuarios (ID, Nombre, Correo, Fecha de registro)."
    },
    # Opcional: puedes agregar 'securityDefinitions' aquí si usas API Keys más adelante.
}
swagger = Swagger(app, template=swagger_template)

# --- DB config (ajusta si hace falta) ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456789",
    "database": "citybike_api",
    "port": 3306,
    "cursorclass": pymysql.cursors.DictCursor
}

def get_connection():
    return pymysql.connect(**DB_CONFIG)

# --- RUTA DE PRUEBA ---
@app.route("/", methods=["GET"])
def home():
    """
    Mensaje de bienvenida
    ---
    responses:
      200:
        description: Mensaje indicando que la API funciona
        examples:
          application/json: {"message": "CityBike API funcionando"}
    """
    return jsonify({"message": "CityBike API funcionando"})

# --- LISTAR USUARIOS ---
@app.route("/usuarios", methods=["GET"])
def get_usuarios():
    """
    Obtener lista de usuarios
    ---
    responses:
      200:
        description: Lista de usuarios
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: integer
              nombre:
                type: string
              correo:
                type: string
              fecha_registro:
                type: string
        examples:
          application/json:
            - id: 1
              nombre: "Ana Torres"
              correo: "ana@example.com"
              fecha_registro: "2025-01-10"
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, nombre, correo, fecha_registro FROM usuarios;")
            rows = cur.fetchall()
        return jsonify(rows), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --- CREAR USUARIO ---
@app.route("/usuarios", methods=["POST"])
def add_usuario():
    """
    Crear un nuevo usuario
    ---
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            nombre:
              type: string
            correo:
              type: string
            fecha_registro:
              type: string
          required:
            - nombre
            - correo
    responses:
      201:
        description: Usuario creado
        schema:
          type: object
          properties:
            message:
              type: string
    """
    data = request.get_json()
    if not data or "nombre" not in data or "correo" not in data:
        return jsonify({"error": "nombre y correo son requeridos"}), 400

    nombre = data.get("nombre")
    correo = data.get("correo")
    fecha = data.get("fecha_registro", None)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO usuarios (nombre, correo, fecha_registro) VALUES (%s, %s, %s)",
                (nombre, correo, fecha)
            )
            conn.commit()
            new_id = cur.lastrowid
        return jsonify({"message": "Usuario creado", "id": new_id}), 201
    except Exception as e:
        # Manejo simple (por ejemplo correo duplicado)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --- ACTUALIZAR USUARIO ---
@app.route("/usuarios/<int:id>", methods=["PUT"])
def update_usuario(id):
    """
    Actualizar un usuario
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            nombre:
              type: string
            correo:
              type: string
    responses:
      200:
        description: Usuario actualizado
      404:
        description: Usuario no encontrado
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body requerido"}), 400

    nombre = data.get("nombre")
    correo = data.get("correo")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM usuarios WHERE id=%s", (id,))
            if not cur.fetchone():
                return jsonify({"error": "Usuario no encontrado"}), 404

            cur.execute(
                "UPDATE usuarios SET nombre=%s, correo=%s WHERE id=%s",
                (nombre, correo, id)
            )
            conn.commit()
        return jsonify({"message": "Usuario actualizado"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --- ELIMINAR USUARIO ---
@app.route("/usuarios/<int:id>", methods=["DELETE"])
def delete_usuario(id):
    """
    Eliminar un usuario
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Usuario eliminado
      404:
        description: Usuario no encontrado
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM usuarios WHERE id=%s", (id,))
            if not cur.fetchone():
                return jsonify({"error": "Usuario no encontrado"}), 404
            cur.execute("DELETE FROM usuarios WHERE id=%s", (id,))
            conn.commit()
        return jsonify({"message": "Usuario eliminado"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
