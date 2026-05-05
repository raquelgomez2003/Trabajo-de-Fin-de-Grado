def validar_float(valor):
    try:
        return float(valor)
    except:
        raise ValueError("Valor inválido")