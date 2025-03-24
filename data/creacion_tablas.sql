CREATE TABLE comunas_provincias_territorio(
  id_comunaint(11) NOT NULL,
  nombre_comunavarchar(100) NOT NULL,
  nombre_provinciavarchar(100) NOT NULL,
  identificador_regionvarchar(30) NOT NULL,
  PRIMARY KEY (id_comuna),
  UNIQUE KEY nombre_comuna(nombre_comuna,id_comuna)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci


CREATE TABLE dimensiones_proyecto (
  id_dim int(11) NOT NULL,
  nombre_dim varchar(100) NOT NULL,
  nombre_completo_dim varchar(100) NOT NULL,
  PRIMARY KEY (id_dim,nombre_dim)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci


CREATE TABLE avance_indicadores (
  id int(11) NOT NULL,
  id_indicador varchar(10) NOT NULL,
  dimension varchar(150) NOT NULL,
  indicador varchar(300) NOT NULL,
  indicador_reformulado varchar(250) DEFAULT NULL,
  estado varchar(20) NOT NULL,
  origen varchar(20) NOT NULL,
  fecha_actualizacion int(11) NOT NULL,
  PRIMARY KEY (id,id_indicador,fecha_actualizacion)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci


CREATE TABLE comunas_provincias_territorio (
  id_comuna int(11) NOT NULL,
  nombre_comuna varchar(100) NOT NULL,
  nombre_provincia varchar(100) NOT NULL,
  identificador_region varchar(30) NOT NULL,
  PRIMARY KEY (id_comuna),
  UNIQUE KEY nombre_comuna (nombre_comuna,id_comuna)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci