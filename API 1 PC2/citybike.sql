-- Crear base de datos
CREATE DATABASE IF NOT EXISTS citybike_api;
USE citybike_api;

-- Crear tabla users
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(150) NOT NULL UNIQUE,
  registration_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insertar 5 registros de ejemplo
INSERT INTO users (name, email, registration_date) VALUES
('Luis Fernández', 'luis.fernandez@example.com', '2025-04-01 09:10:00'),
('María Gómez',     'maria.gomez@example.com',     '2025-04-02 11:35:00'),
('Carlos Paredes',  'carlos.paredes@example.com',  '2025-04-03 14:20:00'),
('Ana Torres',      'ana.torres@example.com',      '2025-04-04 08:05:00'),
('Diego Rojas',     'diego.rojas@example.com',     '2025-04-05 17:55:00');

ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456789';
FLUSH PRIVILEGES;

SELECT user, host, plugin FROM mysql.user;

