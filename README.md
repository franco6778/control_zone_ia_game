# Control Zones AI

Juego 2D de control de zonas en un mapa rectangular grande, basado en `sumo_ai` (algoritmos genéticos + redes neuronales + REINFORCE + Hall of Fame).

## Reglas

- **Mapa rectangular** con paredes sólidas (los robots rebotan en los bordes).
- **5 zonas de control pequeñas** (`zone_radius = 42`) que se van desbloqueando progresivamente cada `12s` (o enseguida si todas las zonas abiertas ya fueron capturadas).
- **10 segundos para capturar**: el robot que permanezca **solo** `10s` dentro de una zona activa la captura y suma 1 punto.
- **Disputa y empuje fuerte**: si ambos robots están dentro de la misma zona, el contador se pausa (*DISPUTA*). Los choques tienen un retroceso fuerte (`knock_base = 175`, `knock_gain = 1.45`) para obligar a empujar al rival fuera de la zona.
- **Victoria**: el primero en capturar **3 de las 5 zonas** gana de inmediato (si se acaba el tiempo, gana quien tenga más zonas o mayor progreso acumulado).

## Uso

```bash
# Entrenar con visualización
./run.sh

# Entrenar sin ventana y guardar métricas en CSV
./run.sh --fresh --headless --generations 50 --log training_log.csv

# Ver demostración del mejor genoma
./run.sh --watch

# Jugar contra la IA (Flechas / WASD)
./run.sh --play

# Evaluar el mejor genoma guardado
./run.sh --eval
```

## Tests

```bash
/home/franco/sumo_ai/.venv/bin/python -m unittest -v test_koth.py
```
