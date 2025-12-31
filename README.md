# Blister Generator Backend

Backend Flask per generare modelli 3D di blister pack in formato STEP.

## Tecnologie
- Python 3.11
- Flask
- CadQuery

## API Endpoint

### POST /convert
Genera un file STEP da parametri JSON.

**Request body:**
```json
{
  "commands": "generate_pallet",
  "parameters": {
    "rows": 2,
    "columns": 3,
    "long_side": 98,
    "short_side": 67,
    ...
  }
}
```

**Response:** File STEP binario

## Deploy
Deployato su Railway.