"""Calculadora y Analizador de Acarreo de Material en Terracerías.

Aplicación web en Streamlit para estimar el número de viajes y el costo del
acarreo de material en obras de movimiento de tierras, comparando varias
capacidades de camión de volteo (análisis heurístico de escenarios).

Proyecto final de Ingeniería Civil (Heurística y Base de Datos).

Ejecución:
    pip install streamlit pandas numpy matplotlib
    streamlit run app.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# CONSTANTES DE INGENIERÍA
# ---------------------------------------------------------------------------
FACTORES_ABUNDAMIENTO = {
    "Tierra común": 1.25,
    "Arcilla compacta": 1.30,
    "Arena / Grava": 1.12,
    "Roca volada": 1.50,
    "Personalizado": None,
}
CAPACIDADES_CAMION_PREDEFINIDAS_M3 = (7.0, 14.0, 20.0, 30.0)
DECIMALES_TOLERANCIA = 9  # evita viajes extra por ruido de punto flotante
EFICIENCIA_MINIMA_RECOMENDADA_PCT = 85.0
FACTOR_ABUNDAMIENTO_MAXIMO_USUAL = 1.60

COL_CAPACIDAD = "Capacidad (m³)"
COL_TARIFA_BASE = "Tarifa base (por viaje)"
COL_TARIFA_KM = "Tarifa por km (por viaje)"
TARIFAS_REFERENCIA_INICIALES = {
    COL_CAPACIDAD: [7.0, 14.0, 20.0, 30.0],
    COL_TARIFA_BASE: [250.0, 400.0, 550.0, 700.0],
    COL_TARIFA_KM: [12.0, 18.0, 24.0, 30.0],
}


# ---------------------------------------------------------------------------
# FUNCIONES DE VALIDACIÓN
# ---------------------------------------------------------------------------
def validar_entradas(
    volumen_banco_m3,
    factor_abundamiento,
    capacidad_camion_m3,
    eficiencia_llenado_pct,
    distancia_km,
    tarifa_base,
    tarifa_por_km,
):
    """Valida que los datos de entrada sean físicamente viables.

    Args:
        volumen_banco_m3 (float): Volumen en banco, en m³.
        factor_abundamiento (float): Factor de abundamiento Fa (adimensional).
        capacidad_camion_m3 (float): Capacidad nominal del camión, en m³ sueltos.
        eficiencia_llenado_pct (float): Eficiencia de llenado, en porcentaje.
        distancia_km (float): Distancia de acarreo, en km.
        tarifa_base (float): Tarifa base por viaje, en moneda local.
        tarifa_por_km (float): Costo adicional por km de acarreo y por viaje.

    Returns:
        list[str]: Advertencias no críticas (puede estar vacía).

    Raises:
        ValueError: Si algún dato es nulo, no numérico, infinito o
            físicamente inviable.
    """
    parametros = {
        "Volumen en banco": volumen_banco_m3,
        "Factor de abundamiento": factor_abundamiento,
        "Capacidad del camión": capacidad_camion_m3,
        "Eficiencia de llenado": eficiencia_llenado_pct,
        "Distancia de acarreo": distancia_km,
        "Tarifa base": tarifa_base,
        "Tarifa por km": tarifa_por_km,
    }
    for nombre, valor in parametros.items():
        if valor is None:
            raise ValueError(f"«{nombre}» está vacío (valor nulo).")
        try:
            valor_numerico = float(valor)
        except (TypeError, ValueError) as error:
            raise ValueError(f"«{nombre}» no es un número válido.") from error
        if not math.isfinite(valor_numerico):
            raise ValueError(f"«{nombre}» debe ser un número finito.")

    if volumen_banco_m3 <= 0:
        raise ValueError("El volumen en banco debe ser mayor que cero.")
    if factor_abundamiento < 1.0:
        raise ValueError(
            "El factor de abundamiento no puede ser menor que 1.0: el material "
            "suelto nunca ocupa menos volumen que en banco."
        )
    if capacidad_camion_m3 <= 0:
        raise ValueError("La capacidad del camión debe ser mayor que cero.")
    if not 0 < eficiencia_llenado_pct <= 100:
        raise ValueError(
            "La eficiencia de llenado debe estar en el intervalo (0, 100] %."
        )
    if distancia_km < 0:
        raise ValueError("La distancia de acarreo no puede ser negativa.")
    if tarifa_base < 0 or tarifa_por_km < 0:
        raise ValueError("Las tarifas no pueden ser negativas.")

    advertencias = []
    if eficiencia_llenado_pct < EFICIENCIA_MINIMA_RECOMENDADA_PCT:
        advertencias.append(
            f"Eficiencia de llenado de {eficiencia_llenado_pct:.0f} %: en obra "
            f"lo usual es entre {EFICIENCIA_MINIMA_RECOMENDADA_PCT:.0f} % y 100 %."
        )
    if factor_abundamiento > FACTOR_ABUNDAMIENTO_MAXIMO_USUAL:
        advertencias.append(
            f"Fa = {factor_abundamiento:.2f} es poco usual "
            f"(típico ≤ {FACTOR_ABUNDAMIENTO_MAXIMO_USUAL:.2f}). Verifica el dato."
        )
    if capacidad_camion_m3 > 40:
        advertencias.append(
            "Capacidad mayor a 40 m³: poco común en camiones de volteo."
        )
    return advertencias


def validar_tarifas_referencia(tarifas_referencia):
    """Valida la tabla editable de tarifas de camiones estándar.

    Args:
        tarifas_referencia (pd.DataFrame): Tabla con capacidad, tarifa base
            y tarifa por km de cada camión estándar.

    Raises:
        ValueError: Si hay celdas vacías, capacidades no positivas, tarifas
            negativas o menos de 3 camiones estándar.
    """
    if tarifas_referencia[[COL_CAPACIDAD, COL_TARIFA_BASE, COL_TARIFA_KM]]\
            .isnull().any().any():
        raise ValueError("La tabla de tarifas de referencia tiene celdas vacías.")
    if (tarifas_referencia[COL_CAPACIDAD] <= 0).any():
        raise ValueError("Las capacidades de referencia deben ser mayores que 0.")
    if (tarifas_referencia[[COL_TARIFA_BASE, COL_TARIFA_KM]] < 0).any().any():
        raise ValueError("Las tarifas de referencia no pueden ser negativas.")
    if len(tarifas_referencia) < 3:
        raise ValueError("Se requieren al menos 3 camiones estándar para comparar.")


# ---------------------------------------------------------------------------
# FUNCIONES DE CÁLCULO (puras)
# ---------------------------------------------------------------------------
def calcular_volumen_suelto(volumen_banco_m3, factor_abundamiento):
    """Calcula el volumen suelto (abundado) del material excavado.

    Vs = Vb × Fa

    Args:
        volumen_banco_m3 (float): Volumen en banco, en m³.
        factor_abundamiento (float): Factor de abundamiento Fa (≥ 1.0).

    Returns:
        float: Volumen suelto total, en m³.

    Raises:
        ValueError: Si el volumen es ≤ 0 o el factor es < 1.0.
    """
    if volumen_banco_m3 <= 0:
        raise ValueError("El volumen en banco debe ser mayor que cero.")
    if factor_abundamiento < 1.0:
        raise ValueError("El factor de abundamiento debe ser ≥ 1.0.")
    return float(volumen_banco_m3 * factor_abundamiento)


def calcular_capacidad_efectiva(capacidad_camion_m3, eficiencia_llenado_pct):
    """Calcula la capacidad efectiva real de carga del camión.

    Cef = Cc × (Ef / 100)

    Args:
        capacidad_camion_m3 (float): Capacidad nominal, en m³ sueltos.
        eficiencia_llenado_pct (float): Eficiencia de llenado, en %.

    Returns:
        float: Capacidad efectiva por viaje, en m³ sueltos.

    Raises:
        ValueError: Si la capacidad es ≤ 0 o la eficiencia no está en (0, 100].
    """
    if capacidad_camion_m3 <= 0:
        raise ValueError("La capacidad del camión debe ser mayor que cero.")
    if not 0 < eficiencia_llenado_pct <= 100:
        raise ValueError("La eficiencia debe estar en el intervalo (0, 100] %.")
    return float(capacidad_camion_m3 * eficiencia_llenado_pct / 100.0)


def calcular_costo_por_viaje(tarifa_base, tarifa_por_km, distancia_km):
    """Calcula el costo de un viaje redondo en función de la distancia.

    Pv = tarifa_base + tarifa_por_km × D

    Args:
        tarifa_base (float): Tarifa base por viaje, en moneda local.
        tarifa_por_km (float): Costo adicional por km, en moneda local/km.
        distancia_km (float): Distancia de acarreo, en km.

    Returns:
        float: Costo por viaje, en moneda local.

    Raises:
        ValueError: Si alguna tarifa o la distancia son negativas.
    """
    if tarifa_base < 0 or tarifa_por_km < 0 or distancia_km < 0:
        raise ValueError("Tarifas y distancia no pueden ser negativas.")
    return float(tarifa_base + tarifa_por_km * distancia_km)


def calcular_viajes_y_costos(
    volumen_suelto_m3, capacidad_efectiva_m3, costo_por_viaje, volumen_banco_m3
):
    """Calcula viajes, costo total, costo unitario y aprovechamiento.

    Nv = ceil(Vs / Cef);  CT = Nv × Pv;  Cub = CT / Vb

    Args:
        volumen_suelto_m3 (float): Volumen suelto total, en m³.
        capacidad_efectiva_m3 (float): Capacidad efectiva por viaje, en m³.
        costo_por_viaje (float): Costo de cada viaje, en moneda local.
        volumen_banco_m3 (float): Volumen en banco, en m³.

    Returns:
        dict: Con las claves ``viajes_exactos`` (float), ``numero_viajes``
        (int), ``costo_total`` (float, moneda local),
        ``costo_unitario_banco`` (float, moneda local/m³),
        ``aprovechamiento_pct`` (float, %) y ``carga_ultimo_viaje_m3``
        (float, m³).

    Raises:
        ValueError: Si la capacidad efectiva o el volumen en banco son ≤ 0.
    """
    if capacidad_efectiva_m3 <= 0:
        raise ValueError("La capacidad efectiva debe ser mayor que cero.")
    if volumen_banco_m3 <= 0:
        raise ValueError("El volumen en banco debe ser mayor que cero.")

    viajes_exactos = volumen_suelto_m3 / capacidad_efectiva_m3
    numero_viajes = math.ceil(round(viajes_exactos, DECIMALES_TOLERANCIA))
    costo_total = numero_viajes * costo_por_viaje
    return {
        "viajes_exactos": viajes_exactos,
        "numero_viajes": numero_viajes,
        "costo_total": costo_total,
        "costo_unitario_banco": costo_total / volumen_banco_m3,
        "aprovechamiento_pct": (
            volumen_suelto_m3 / (numero_viajes * capacidad_efectiva_m3) * 100.0
        ),
        "carga_ultimo_viaje_m3": (
            volumen_suelto_m3 - (numero_viajes - 1) * capacidad_efectiva_m3
        ),
    }


def nombres_columnas(moneda):
    """Devuelve los encabezados (con unidades) de las tablas de resultados.

    Args:
        moneda (str): Código de moneda, por ejemplo "MXN" o "USD".

    Returns:
        dict[str, str]: Clave lógica → encabezado de columna.
    """
    return {
        "escenario": "Escenario",
        "capacidad": "Capacidad nominal (m³)",
        "capacidad_efectiva": "Capacidad efectiva (m³)",
        "volumen_banco": "Volumen en banco (m³)",
        "volumen_suelto": "Volumen suelto (m³)",
        "viajes": "Núm. de viajes",
        "costo_viaje": f"Costo por viaje ({moneda})",
        "costo_total": f"Costo total ({moneda})",
        "costo_unitario": f"Costo por m³ en banco ({moneda}/m³)",
        "aprovechamiento": "Aprovechamiento (%)",
        "ranking": "Ranking",
    }


def construir_escenarios(
    capacidad_seleccionada_m3,
    tarifa_base_seleccionada,
    tarifa_km_seleccionada,
    tarifas_referencia,
):
    """Arma la lista de escenarios: camión seleccionado + camiones estándar.

    Si la capacidad seleccionada coincide con una estándar, se usa la tarifa
    ingresada por el usuario y no se duplica el escenario.

    Args:
        capacidad_seleccionada_m3 (float): Capacidad del camión elegido, m³.
        tarifa_base_seleccionada (float): Tarifa base del camión elegido.
        tarifa_km_seleccionada (float): Tarifa por km del camión elegido.
        tarifas_referencia (pd.DataFrame): Tarifas de camiones estándar.

    Returns:
        dict: Con ``capacidades`` (np.ndarray, m³), ``tarifas_base``
        (np.ndarray), ``tarifas_km`` (np.ndarray) y ``etiquetas``
        (list[str]); el índice 0 es siempre el camión seleccionado.

    Raises:
        ValueError: Si la tabla de referencia no es válida.
    """
    validar_tarifas_referencia(tarifas_referencia)
    referencia = tarifas_referencia[
        ~np.isclose(tarifas_referencia[COL_CAPACIDAD], capacidad_seleccionada_m3)
    ]
    capacidades_estandar = referencia[COL_CAPACIDAD].to_numpy(dtype=float)
    return {
        "capacidades": np.concatenate(
            ([capacidad_seleccionada_m3], capacidades_estandar)
        ),
        "tarifas_base": np.concatenate(
            ([tarifa_base_seleccionada],
             referencia[COL_TARIFA_BASE].to_numpy(dtype=float))
        ),
        "tarifas_km": np.concatenate(
            ([tarifa_km_seleccionada],
             referencia[COL_TARIFA_KM].to_numpy(dtype=float))
        ),
        "etiquetas": [f"{capacidad_seleccionada_m3:g} m³ (sel.)"]
        + [f"{capacidad:g} m³" for capacidad in capacidades_estandar],
    }


def generar_tabla_comparativa(
    volumen_banco_m3,
    factor_abundamiento,
    eficiencia_llenado_pct,
    distancia_km,
    escenarios,
    moneda,
):
    """Genera la tabla comparativa de camiones con cálculo vectorizado (NumPy).

    Ordena los escenarios por costo total y número de viajes (heurística de
    optimización: gana el más económico; a igual costo, el de menos viajes).

    Args:
        volumen_banco_m3 (float): Volumen en banco, en m³.
        factor_abundamiento (float): Factor de abundamiento (≥ 1.0).
        eficiencia_llenado_pct (float): Eficiencia de llenado, en %.
        distancia_km (float): Distancia de acarreo, en km.
        escenarios (dict): Salida de :func:`construir_escenarios`.
        moneda (str): Código de moneda para los encabezados.

    Returns:
        pd.DataFrame: Una fila por escenario, ordenada por ranking.

    Raises:
        ValueError: Si algún dato de entrada es físicamente inviable.
    """
    capacidades = np.asarray(escenarios["capacidades"], dtype=float)
    if np.any(capacidades <= 0):
        raise ValueError("Todas las capacidades de camión deben ser > 0.")

    volumen_suelto = calcular_volumen_suelto(volumen_banco_m3, factor_abundamiento)
    capacidades_efectivas = capacidades * eficiencia_llenado_pct / 100.0
    numero_viajes = np.ceil(
        np.round(volumen_suelto / capacidades_efectivas, DECIMALES_TOLERANCIA)
    )
    costos_por_viaje = (
        escenarios["tarifas_base"] + escenarios["tarifas_km"] * distancia_km
    )
    costos_totales = numero_viajes * costos_por_viaje
    col = nombres_columnas(moneda)

    tabla = pd.DataFrame(
        {
            col["escenario"]: escenarios["etiquetas"],
            col["capacidad"]: capacidades,
            col["capacidad_efectiva"]: capacidades_efectivas,
            col["volumen_banco"]: volumen_banco_m3,
            col["volumen_suelto"]: volumen_suelto,
            col["viajes"]: numero_viajes.astype(int),
            col["costo_viaje"]: costos_por_viaje,
            col["costo_total"]: costos_totales,
            col["costo_unitario"]: costos_totales / volumen_banco_m3,
            col["aprovechamiento"]: (
                volumen_suelto / (numero_viajes * capacidades_efectivas) * 100.0
            ),
        }
    )
    tabla = tabla.sort_values([col["costo_total"], col["viajes"]])
    tabla = tabla.reset_index(drop=True)
    tabla[col["ranking"]] = np.arange(1, len(tabla) + 1)
    return tabla


def generar_tabla_materiales(
    volumen_banco_m3,
    factor_personalizado,
    capacidad_efectiva_m3,
    costo_por_viaje,
    moneda,
):
    """Compara los tipos de material usando el camión seleccionado.

    Args:
        volumen_banco_m3 (float): Volumen en banco, en m³.
        factor_personalizado (float | None): Fa del material personalizado, o
            ``None`` si no se está usando.
        capacidad_efectiva_m3 (float): Capacidad efectiva del camión, en m³.
        costo_por_viaje (float): Costo por viaje, en moneda local.
        moneda (str): Código de moneda para los encabezados.

    Returns:
        pd.DataFrame: Una fila por material con Fa, volúmenes, viajes y costos.

    Raises:
        ValueError: Si el volumen o la capacidad efectiva son ≤ 0.
    """
    if volumen_banco_m3 <= 0 or capacidad_efectiva_m3 <= 0:
        raise ValueError("Volumen y capacidad efectiva deben ser mayores que 0.")

    nombres = [m for m, fa in FACTORES_ABUNDAMIENTO.items() if fa is not None]
    factores = [FACTORES_ABUNDAMIENTO[m] for m in nombres]
    if factor_personalizado is not None:
        nombres.append(f"Personalizado (Fa={factor_personalizado:.2f})")
        factores.append(factor_personalizado)

    factores = np.asarray(factores, dtype=float)
    volumenes_sueltos = volumen_banco_m3 * factores
    numero_viajes = np.ceil(
        np.round(volumenes_sueltos / capacidad_efectiva_m3, DECIMALES_TOLERANCIA)
    )
    costos_totales = numero_viajes * costo_por_viaje
    col = nombres_columnas(moneda)
    return pd.DataFrame(
        {
            "Material": nombres,
            "Fa": factores,
            col["volumen_banco"]: volumen_banco_m3,
            col["volumen_suelto"]: volumenes_sueltos,
            col["viajes"]: numero_viajes.astype(int),
            col["costo_total"]: costos_totales,
            col["costo_unitario"]: costos_totales / volumen_banco_m3,
        }
    )


def calcular_curvas_volumen(volumenes_banco, factor, capacidades_efectivas,
                            costos_por_viaje):
    """Calcula el costo total en función del volumen (curva escalonada).

    Args:
        volumenes_banco (np.ndarray): Volúmenes en banco a evaluar, en m³.
        factor (float): Factor de abundamiento (≥ 1.0).
        capacidades_efectivas (np.ndarray): Capacidad efectiva por escenario, m³.
        costos_por_viaje (np.ndarray): Costo por viaje de cada escenario.

    Returns:
        np.ndarray: Matriz (escenarios × volúmenes) con el costo total.
    """
    viajes = np.ceil(
        np.round(
            volumenes_banco[None, :] * factor / capacidades_efectivas[:, None],
            DECIMALES_TOLERANCIA,
        )
    )
    return viajes * costos_por_viaje[:, None]


def calcular_curvas_distancia(distancias_km, numero_viajes, tarifas_base,
                              tarifas_km):
    """Calcula el costo total en función de la distancia de acarreo.

    Args:
        distancias_km (np.ndarray): Distancias a evaluar, en km.
        numero_viajes (np.ndarray): Viajes requeridos por escenario.
        tarifas_base (np.ndarray): Tarifa base por viaje de cada escenario.
        tarifas_km (np.ndarray): Tarifa por km de cada escenario.

    Returns:
        np.ndarray: Matriz (escenarios × distancias) con el costo total.
    """
    costos_por_viaje = (
        tarifas_base[:, None] + tarifas_km[:, None] * distancias_km[None, :]
    )
    return numero_viajes[:, None] * costos_por_viaje


# ---------------------------------------------------------------------------
# FUNCIONES DE GRÁFICOS (Matplotlib)
# ---------------------------------------------------------------------------
def graficar_analisis(tabla_camiones, tabla_materiales, moneda):
    """Gráfico 1: volumen en banco vs. suelto y costo total por camión.

    Args:
        tabla_camiones (pd.DataFrame): Salida de :func:`generar_tabla_comparativa`.
        tabla_materiales (pd.DataFrame): Salida de :func:`generar_tabla_materiales`.
        moneda (str): Código de moneda para los ejes.

    Returns:
        matplotlib.figure.Figure: Figura con dos paneles de barras.
    """
    col = nombres_columnas(moneda)
    figura, (eje_volumen, eje_costo) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel izquierdo: abundamiento por tipo de material
    posiciones = np.arange(len(tabla_materiales))
    ancho_barra = 0.38
    barras_banco = eje_volumen.bar(
        posiciones - ancho_barra / 2, tabla_materiales[col["volumen_banco"]],
        ancho_barra, color="#6c757d", label="Volumen en banco",
    )
    barras_suelto = eje_volumen.bar(
        posiciones + ancho_barra / 2, tabla_materiales[col["volumen_suelto"]],
        ancho_barra, color="#e8833a", label="Volumen suelto abundado",
    )
    eje_volumen.bar_label(barras_banco, fmt="%.0f", padding=2, fontsize=8)
    eje_volumen.bar_label(barras_suelto, fmt="%.0f", padding=2, fontsize=8)
    eje_volumen.set_xticks(posiciones)
    eje_volumen.set_xticklabels(tabla_materiales["Material"], rotation=20,
                                ha="right", fontsize=9)
    eje_volumen.set_title("Volumen en banco vs. volumen suelto por material")
    eje_volumen.set_xlabel("Tipo de material")
    eje_volumen.set_ylabel(r"Volumen ($\mathrm{m^3}$)")
    eje_volumen.set_axisbelow(True)
    eje_volumen.grid(True, linestyle="--", alpha=0.5)
    eje_volumen.legend()

    # Panel derecho: costo total por capacidad de camión
    datos = tabla_camiones.sort_values(col["capacidad"])
    posiciones_camion = np.arange(len(datos))
    es_mejor = (datos[col["ranking"]] == 1).to_numpy()
    colores = ["#2e8b57" if mejor else "#3b6ea5" for mejor in es_mejor]
    barras_costo = eje_costo.bar(posiciones_camion, datos[col["costo_total"]],
                                 color=colores)
    eje_costo.bar_label(
        barras_costo,
        labels=[f"{n} viajes" for n in datos[col["viajes"]]],
        padding=2, fontsize=8,
    )
    eje_costo.set_xticks(posiciones_camion)
    eje_costo.set_xticklabels(datos[col["escenario"]], fontsize=9)
    eje_costo.set_title("Costo total por capacidad de camión")
    eje_costo.set_xlabel(r"Capacidad del camión ($\mathrm{m^3}$)")
    eje_costo.set_ylabel(f"Costo total ({moneda})")
    eje_costo.set_axisbelow(True)
    eje_costo.grid(True, linestyle="--", alpha=0.5)
    eje_costo.legend(handles=[
        Patch(color="#2e8b57", label="Opción más económica"),
        Patch(color="#3b6ea5", label="Otras opciones"),
    ])
    figura.tight_layout()
    return figura


def _dibujar_curvas(eje, valores_x, curvas, etiquetas, punto_actual, etiqueta_x,
                    titulo, moneda):
    """Dibuja un conjunto de curvas de costo sobre un eje de Matplotlib.

    Args:
        eje (matplotlib.axes.Axes): Eje donde se dibuja.
        valores_x (np.ndarray): Valores de la variable independiente.
        curvas (np.ndarray): Matriz (escenarios × puntos) de costo total.
        etiquetas (list[str]): Nombre de cada escenario (índice 0 = seleccionado).
        punto_actual (tuple[float, float]): Punto (x, costo) de diseño.
        etiqueta_x (str): Texto del eje x con sus unidades.
        titulo (str): Título del gráfico.
        moneda (str): Código de moneda para el eje y.
    """
    for indice, (curva, etiqueta) in enumerate(zip(curvas, etiquetas)):
        eje.plot(valores_x, curva, label=etiqueta,
                 linewidth=2.6 if indice == 0 else 1.4,
                 alpha=1.0 if indice == 0 else 0.8)
    eje.scatter(*punto_actual, color="red", zorder=5, s=60,
                label="Punto de diseño")
    eje.set_title(titulo)
    eje.set_xlabel(etiqueta_x)
    eje.set_ylabel(f"Costo total ({moneda})")
    eje.grid(True, linestyle="--", alpha=0.5)
    eje.legend(fontsize=8)


def graficar_sensibilidad_volumen(
    escenarios, factor, eficiencia_llenado_pct, volumen_banco_m3, distancia_km,
    moneda,
):
    """Gráfico 2: curva de sensibilidad Costo total vs. volumen en banco.

    El panel derecho amplía el entorno del volumen de diseño para apreciar el
    comportamiento escalonado (un viaje adicional = un escalón de costo).

    Args:
        escenarios (dict): Salida de :func:`construir_escenarios`.
        factor (float): Factor de abundamiento (≥ 1.0).
        eficiencia_llenado_pct (float): Eficiencia de llenado, en %.
        volumen_banco_m3 (float): Volumen de diseño en banco, en m³.
        distancia_km (float): Distancia de acarreo, en km.
        moneda (str): Código de moneda para los ejes.

    Returns:
        matplotlib.figure.Figure: Figura con la curva completa y su ampliación.
    """
    capacidades_efectivas = escenarios["capacidades"] * eficiencia_llenado_pct / 100
    costos_por_viaje = escenarios["tarifas_base"] + escenarios["tarifas_km"] * \
        distancia_km
    costo_actual = calcular_curvas_volumen(
        np.array([volumen_banco_m3]), factor, capacidades_efectivas,
        costos_por_viaje,
    )[0, 0]

    volumenes_completos = np.linspace(0.02 * volumen_banco_m3,
                                      1.6 * volumen_banco_m3, 4000)
    volumenes_zoom = np.linspace(0.9 * volumen_banco_m3,
                                 1.1 * volumen_banco_m3, 2000)
    figura, (eje_completo, eje_zoom) = plt.subplots(1, 2, figsize=(14, 5.5))
    etiqueta_x = r"Volumen en banco ($\mathrm{m^3}$)"
    for eje, volumenes, titulo in (
        (eje_completo, volumenes_completos,
         "Sensibilidad: costo total vs. volumen"),
        (eje_zoom, volumenes_zoom,
         "Ampliación (±10 %): escalones por viaje adicional"),
    ):
        curvas = calcular_curvas_volumen(volumenes, factor,
                                         capacidades_efectivas,
                                         costos_por_viaje)
        _dibujar_curvas(eje, volumenes, curvas, escenarios["etiquetas"],
                        (volumen_banco_m3, costo_actual), etiqueta_x, titulo,
                        moneda)
    figura.tight_layout()
    return figura


def graficar_sensibilidad_distancia(
    escenarios, numero_viajes, distancia_km, moneda
):
    """Gráfico 3: curva de sensibilidad Costo total vs. distancia de acarreo.

    Args:
        escenarios (dict): Salida de :func:`construir_escenarios`.
        numero_viajes (np.ndarray): Viajes requeridos por escenario.
        distancia_km (float): Distancia de diseño, en km.
        moneda (str): Código de moneda para los ejes.

    Returns:
        matplotlib.figure.Figure: Figura con las curvas de costo vs. distancia.
    """
    distancias = np.linspace(0, max(2 * distancia_km, 20.0), 200)
    curvas = calcular_curvas_distancia(
        distancias, numero_viajes, escenarios["tarifas_base"],
        escenarios["tarifas_km"],
    )
    costo_actual = calcular_curvas_distancia(
        np.array([distancia_km]), numero_viajes, escenarios["tarifas_base"],
        escenarios["tarifas_km"],
    )[0, 0]
    figura, eje = plt.subplots(figsize=(8, 5))
    _dibujar_curvas(
        eje, distancias, curvas, escenarios["etiquetas"],
        (distancia_km, costo_actual), "Distancia de acarreo ($\\mathrm{km}$)",
        "Sensibilidad: costo total vs. distancia de acarreo", moneda,
    )
    figura.tight_layout()
    return figura


# ---------------------------------------------------------------------------
# INTERFAZ (Streamlit)
# ---------------------------------------------------------------------------
def solicitar_entradas_usuario():
    """Construye la barra lateral y recoge los datos de entrada.

    Returns:
        dict: Datos ingresados: material, factor de abundamiento (Fa),
        volumen en banco (m³), capacidad del camión (m³), eficiencia (%),
        distancia (km), tarifas, moneda y tabla de tarifas de referencia.
    """
    st.sidebar.header("⚙️ Parámetros de entrada")

    material = st.sidebar.selectbox("Tipo de material", list(FACTORES_ABUNDAMIENTO))
    if FACTORES_ABUNDAMIENTO[material] is None:
        factor = st.sidebar.number_input(
            "Factor de abundamiento Fa (personalizado)", value=1.25, step=0.01,
            format="%.2f",
        )
    else:
        factor = FACTORES_ABUNDAMIENTO[material]
        st.sidebar.caption(f"Fa = {factor:.2f} (valor típico de ingeniería)")

    volumen_banco = st.sidebar.number_input(
        "Volumen en banco Vb (m³)", value=1500.0, step=50.0, format="%.2f",
    )

    opciones_capacidad = [f"{c:g}" for c in CAPACIDADES_CAMION_PREDEFINIDAS_M3]
    opcion = st.sidebar.selectbox(
        "Capacidad del camión Cc (m³ sueltos)",
        opciones_capacidad + ["Personalizada"], index=1,
    )
    if opcion == "Personalizada":
        capacidad = st.sidebar.number_input(
            "Capacidad personalizada (m³)", value=10.0, step=0.5, format="%.2f",
        )
    else:
        capacidad = float(opcion)

    eficiencia = st.sidebar.slider(
        "Eficiencia de llenado Ef (%)", min_value=50, max_value=100, value=90,
    )
    distancia = st.sidebar.number_input(
        "Distancia de acarreo D (km)", value=10.0, step=0.5, format="%.2f",
    )
    moneda = st.sidebar.selectbox("Moneda", ["MXN", "USD"])
    tarifa_base = st.sidebar.number_input(
        f"Tarifa base por viaje ({moneda})", value=400.0, step=10.0,
        format="%.2f",
    )
    tarifa_km = st.sidebar.number_input(
        f"Costo adicional por km, por viaje ({moneda}/km)", value=18.0,
        step=1.0, format="%.2f",
    )

    with st.sidebar.expander("Tarifas de camiones estándar (editable)"):
        st.caption(
            "Se usan en el análisis de escenarios. Ajusta con las tarifas "
            "reales de tu región."
        )
        tarifas_referencia = st.data_editor(
            pd.DataFrame(TARIFAS_REFERENCIA_INICIALES),
            disabled=[COL_CAPACIDAD], hide_index=True, key="tarifas_ref",
        )

    return {
        "material": material,
        "factor_abundamiento": factor,
        "volumen_banco_m3": volumen_banco,
        "capacidad_camion_m3": capacidad,
        "eficiencia_llenado_pct": float(eficiencia),
        "distancia_km": distancia,
        "moneda": moneda,
        "tarifa_base": tarifa_base,
        "tarifa_por_km": tarifa_km,
        "tarifas_referencia": tarifas_referencia,
    }


def _resaltar_mejor_opcion(fila, columna_ranking):
    """Pinta de verde la fila de la opción con ranking 1.

    Args:
        fila (pd.Series): Fila de la tabla comparativa.
        columna_ranking (str): Nombre de la columna de ranking.

    Returns:
        list[str]: Estilos CSS, uno por celda de la fila.
    """
    estilo = "background-color: #d4edda" if fila[columna_ranking] == 1 else ""
    return [estilo] * len(fila)


def _latex_numero(valor, decimales=2):
    """Formatea un número para LaTeX usando coma como separador de miles.

    Args:
        valor (float): Número a formatear.
        decimales (int): Cantidad de decimales.

    Returns:
        str: Cadena lista para insertar en una fórmula LaTeX.
    """
    return f"{valor:,.{decimales}f}".replace(",", "{,}")


def mostrar_tab_principal(entradas, resultados, tabla_camiones):
    """Muestra la pestaña 1: KPIs y conclusiones del cálculo principal.

    Args:
        entradas (dict): Datos ingresados por el usuario.
        resultados (dict): Resultados del cálculo del camión seleccionado.
        tabla_camiones (pd.DataFrame): Tabla comparativa ordenada por ranking.
    """
    moneda = entradas["moneda"]
    col = nombres_columnas(moneda)
    st.subheader("Resultados del camión seleccionado")

    primera_fila = st.columns(4)
    primera_fila[0].metric("Volumen suelto Vs",
                           f"{resultados['volumen_suelto']:,.2f} m³")
    primera_fila[1].metric("Total de viajes Nv",
                           f"{resultados['numero_viajes']:,}")
    primera_fila[2].metric("Costo total CT",
                           f"{resultados['costo_total']:,.2f} {moneda}")
    primera_fila[3].metric("Costo unitario en banco",
                           f"{resultados['costo_unitario_banco']:,.2f} {moneda}/m³")

    segunda_fila = st.columns(4)
    segunda_fila[0].metric("Capacidad efectiva Cef",
                           f"{resultados['capacidad_efectiva']:,.2f} m³")
    segunda_fila[1].metric("Costo por viaje Pv",
                           f"{resultados['costo_por_viaje']:,.2f} {moneda}")
    segunda_fila[2].metric("Viajes exactos (sin redondear)",
                           f"{resultados['viajes_exactos']:,.3f}")
    segunda_fila[3].metric("Aprovechamiento de flota",
                           f"{resultados['aprovechamiento_pct']:,.1f} %")

    st.info(
        f"Se necesitan {resultados['viajes_exactos']:,.3f} viajes, pero en obra "
        f"no existen fracciones de viaje: se redondea hacia arriba a "
        f"**{resultados['numero_viajes']} viajes**. El último viaje lleva "
        f"{resultados['carga_ultimo_viaje_m3']:,.2f} m³ de "
        f"{resultados['capacidad_efectiva']:,.2f} m³ posibles."
    )

    mejor = tabla_camiones.iloc[0]
    seleccionado = tabla_camiones[
        tabla_camiones[col["escenario"]].str.contains("sel.", regex=False)
    ].iloc[0]
    ahorro = seleccionado[col["costo_total"]] - mejor[col["costo_total"]]
    if ahorro > 0.005:
        st.success(
            f"🏆 Heurística de optimización: el camión de "
            f"**{mejor[col['capacidad']]:g} m³** es la opción más económica "
            f"({mejor[col['costo_total']]:,.2f} {moneda}). Ahorra "
            f"{ahorro:,.2f} {moneda} frente al camión seleccionado."
        )
    else:
        st.success("🏆 El camión seleccionado ya es la opción más económica "
                   "entre los escenarios analizados.")


def mostrar_tab_comparativo(entradas, tabla_camiones, tabla_materiales,
                            escenarios, numero_viajes):
    """Muestra la pestaña 2: tablas, descarga CSV y gráficos.

    Args:
        entradas (dict): Datos ingresados por el usuario.
        tabla_camiones (pd.DataFrame): Comparativa de camiones.
        tabla_materiales (pd.DataFrame): Comparativa de materiales.
        escenarios (dict): Escenarios construidos (camiones y tarifas).
        numero_viajes (np.ndarray): Viajes requeridos por escenario.
    """
    moneda = entradas["moneda"]
    col = nombres_columnas(moneda)

    st.subheader("Comparativa de camiones (verde = más económico)")
    formatos = {
        col["capacidad"]: "{:,.1f}", col["capacidad_efectiva"]: "{:,.2f}",
        col["volumen_banco"]: "{:,.2f}", col["volumen_suelto"]: "{:,.2f}",
        col["costo_viaje"]: "{:,.2f}", col["costo_total"]: "{:,.2f}",
        col["costo_unitario"]: "{:,.2f}", col["aprovechamiento"]: "{:,.1f}",
    }
    st.dataframe(
        tabla_camiones.style.apply(
            _resaltar_mejor_opcion, axis=1, columna_ranking=col["ranking"]
        ).format(formatos),
        hide_index=True,
    )
    st.download_button(
        "⬇️ Descargar tabla (CSV)",
        data=tabla_camiones.to_csv(index=False).encode("utf-8-sig"),
        file_name="comparativo_acarreo.csv", mime="text/csv",
    )

    st.subheader("Comparativa por tipo de material (camión seleccionado)")
    st.dataframe(
        tabla_materiales.style.format({
            "Fa": "{:.2f}", col["volumen_banco"]: "{:,.2f}",
            col["volumen_suelto"]: "{:,.2f}", col["costo_total"]: "{:,.2f}",
            col["costo_unitario"]: "{:,.2f}",
        }),
        hide_index=True,
    )

    st.subheader("Gráficos de análisis")
    figura_barras = graficar_analisis(tabla_camiones, tabla_materiales, moneda)
    st.pyplot(figura_barras)
    plt.close(figura_barras)

    figura_volumen = graficar_sensibilidad_volumen(
        escenarios, entradas["factor_abundamiento"],
        entradas["eficiencia_llenado_pct"], entradas["volumen_banco_m3"],
        entradas["distancia_km"], moneda,
    )
    st.pyplot(figura_volumen)
    plt.close(figura_volumen)

    figura_distancia = graficar_sensibilidad_distancia(
        escenarios, numero_viajes, entradas["distancia_km"], moneda
    )
    st.pyplot(figura_distancia)
    plt.close(figura_distancia)


def mostrar_memoria_calculo(entradas, resultados):
    """Muestra la pestaña 3: memoria de cálculo con ecuaciones en LaTeX.

    Args:
        entradas (dict): Datos ingresados por el usuario.
        resultados (dict): Resultados del cálculo del camión seleccionado.
    """
    moneda = entradas["moneda"]
    volumen_banco = entradas["volumen_banco_m3"]
    factor = entradas["factor_abundamiento"]
    capacidad = entradas["capacidad_camion_m3"]
    eficiencia = entradas["eficiencia_llenado_pct"]
    distancia = entradas["distancia_km"]
    tarifa_base = entradas["tarifa_base"]
    tarifa_km = entradas["tarifa_por_km"]
    n = _latex_numero

    st.subheader("Memoria de cálculo paso a paso")

    st.markdown("**Paso 1. Volumen suelto (abundamiento del material)**")
    st.latex(
        rf"V_s = V_b \times F_a = {n(volumen_banco)} \times {n(factor)} = "
        rf"{n(resultados['volumen_suelto'])}\ \mathrm{{m^3}}"
    )

    st.markdown("**Paso 2. Capacidad efectiva del camión**")
    st.latex(
        rf"C_{{ef}} = C_c \times \frac{{E_f}}{{100}} = {n(capacidad)} \times "
        rf"\frac{{{n(eficiencia, 0)}}}{{100}} = "
        rf"{n(resultados['capacidad_efectiva'])}\ \mathrm{{m^3}}"
    )

    st.markdown("**Paso 3. Número de viajes (redondeo hacia arriba)**")
    st.latex(
        rf"N_v = \left\lceil \frac{{V_s}}{{C_{{ef}}}} \right\rceil = "
        rf"\left\lceil \frac{{{n(resultados['volumen_suelto'])}}}"
        rf"{{{n(resultados['capacidad_efectiva'])}}} \right\rceil = "
        rf"\lceil {n(resultados['viajes_exactos'], 3)} \rceil = "
        rf"{resultados['numero_viajes']}\ \text{{viajes}}"
    )
    st.caption("No existen fracciones de viaje en obra, por eso se usa la "
               "función techo (math.ceil).")

    st.markdown("**Paso 4. Costo por viaje según la distancia**")
    st.latex(
        rf"P_v = P_{{base}} + P_{{km}} \times D = {n(tarifa_base)} + "
        rf"{n(tarifa_km)} \times {n(distancia)} = "
        rf"{n(resultados['costo_por_viaje'])}\ \mathrm{{{moneda}}}"
    )

    st.markdown("**Paso 5. Costo total del acarreo**")
    st.latex(
        rf"C_T = N_v \times P_v = {resultados['numero_viajes']} \times "
        rf"{n(resultados['costo_por_viaje'])} = "
        rf"{n(resultados['costo_total'])}\ \mathrm{{{moneda}}}"
    )

    st.markdown("**Paso 6. Costo unitario por m³ en banco (precio unitario)**")
    st.latex(
        rf"C_{{ub}} = \frac{{C_T}}{{V_b}} = "
        rf"\frac{{{n(resultados['costo_total'])}}}{{{n(volumen_banco)}}} = "
        rf"{n(resultados['costo_unitario_banco'])}\ "
        rf"\mathrm{{{moneda}/m^3}}"
    )

    st.markdown("**Paso 7. Aprovechamiento de la capacidad contratada**")
    st.latex(
        rf"\eta = \frac{{V_s}}{{N_v \times C_{{ef}}}} \times 100 = "
        rf"\frac{{{n(resultados['volumen_suelto'])}}}"
        rf"{{{resultados['numero_viajes']} \times "
        rf"{n(resultados['capacidad_efectiva'])}}} \times 100 = "
        rf"{n(resultados['aprovechamiento_pct'], 1)}\ \%"
    )


def main():
    """Punto de entrada: configura la página, valida, calcula y muestra."""
    st.set_page_config(
        page_title="Acarreo de Material en Terracerías",
        page_icon="🚚", layout="wide",
    )
    st.title("🚚 Calculadora y Analizador de Acarreo de Material en Terracerías")
    st.caption("Ingeniería Civil · Movimiento de tierras · Análisis comparativo "
               "de camiones de volteo")

    entradas = solicitar_entradas_usuario()
    try:
        advertencias = validar_entradas(
            entradas["volumen_banco_m3"], entradas["factor_abundamiento"],
            entradas["capacidad_camion_m3"], entradas["eficiencia_llenado_pct"],
            entradas["distancia_km"], entradas["tarifa_base"],
            entradas["tarifa_por_km"],
        )
        volumen_suelto = calcular_volumen_suelto(
            entradas["volumen_banco_m3"], entradas["factor_abundamiento"]
        )
        capacidad_efectiva = calcular_capacidad_efectiva(
            entradas["capacidad_camion_m3"], entradas["eficiencia_llenado_pct"]
        )
        costo_por_viaje = calcular_costo_por_viaje(
            entradas["tarifa_base"], entradas["tarifa_por_km"],
            entradas["distancia_km"],
        )
        resultados = calcular_viajes_y_costos(
            volumen_suelto, capacidad_efectiva, costo_por_viaje,
            entradas["volumen_banco_m3"],
        )
        resultados.update({
            "volumen_suelto": volumen_suelto,
            "capacidad_efectiva": capacidad_efectiva,
            "costo_por_viaje": costo_por_viaje,
        })

        escenarios = construir_escenarios(
            entradas["capacidad_camion_m3"], entradas["tarifa_base"],
            entradas["tarifa_por_km"], entradas["tarifas_referencia"],
        )
        tabla_camiones = generar_tabla_comparativa(
            entradas["volumen_banco_m3"], entradas["factor_abundamiento"],
            entradas["eficiencia_llenado_pct"], entradas["distancia_km"],
            escenarios, entradas["moneda"],
        )
        factor_personalizado = (
            entradas["factor_abundamiento"]
            if FACTORES_ABUNDAMIENTO[entradas["material"]] is None else None
        )
        tabla_materiales = generar_tabla_materiales(
            entradas["volumen_banco_m3"], factor_personalizado,
            capacidad_efectiva, costo_por_viaje, entradas["moneda"],
        )
        capacidades_efectivas = (
            escenarios["capacidades"] * entradas["eficiencia_llenado_pct"] / 100
        )
        numero_viajes = np.ceil(
            np.round(volumen_suelto / capacidades_efectivas, DECIMALES_TOLERANCIA)
        )
    except ValueError as error:
        st.error(f"❌ Datos inválidos: {error}")
        st.stop()
    except Exception as error:  # noqa: BLE001 - última barrera de la interfaz
        st.error(f"❌ Error inesperado durante el cálculo: {error}")
        st.stop()

    for advertencia in advertencias:
        st.warning(f"⚠️ {advertencia}")

    pestana_principal, pestana_comparativa, pestana_memoria = st.tabs(
        ["📊 Cálculo y KPIs", "📈 Análisis comparativo", "🧮 Memoria de cálculo"]
    )
    with pestana_principal:
        mostrar_tab_principal(entradas, resultados, tabla_camiones)
    with pestana_comparativa:
        mostrar_tab_comparativo(entradas, tabla_camiones, tabla_materiales,
                                escenarios, numero_viajes)
    with pestana_memoria:
        mostrar_memoria_calculo(entradas, resultados)


if __name__ == "__main__":
    main()
