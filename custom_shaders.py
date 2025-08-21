import numpy as np

def HalftoneToonShader(**kw):
    """Posterizado + patrón de puntos (halftone) en zonas medias."""
    A, B, C = kw["verts"]
    bu, bv, bw = kw["baryCoords"]
    L = np.array(kw["dirLight"], dtype=float)

    # Normal + difusa
    NA, NB, NC = np.array(A[3:6]), np.array(B[3:6]), np.array(C[3:6])
    N = bu*NA + bv*NB + bw*NC;  N /= np.linalg.norm(N)+1e-8
    diff = max(0, N.dot(-L))

    # Posterización de la intensidad
    levels = kw.get("levels", 4)
    diff_q = np.floor(diff * levels) / (levels-1)

    # Color base (desde textura o constante)
    if (T:=kw.get("texture")) is not None:
        r, g, b = T.getColor(kw.get("u"), kw.get("v"))
    else:
        r, g, b = kw.get("baseColor", (1,1,1))

    color = np.array((r, g, b)) * (kw.get("ambient",0.2) + diff_q*(1-kw.get("ambient",0.2)))

    # Halftone pattern — usa coords de pantalla si están en kw
    x_pix = kw.get("x", 0); y_pix = kw.get("y", 0)
    dot_size = kw.get("dotSize", 4)
    mask = ((x_pix//dot_size)+(y_pix//dot_size))%2  # tablero
    mid_band = diff_q > 0.3 and diff_q < 0.7
    if mid_band and mask:
        color *= 0.5  # oscurece puntos alternos

    return tuple((color*255).astype(int))


# 4. WaveVertexShader (Vertex)  — reemplaza Twist


def WaveVertexShader(vertex, **kw):
    """Ondea la malla como si viajara una ola senoidal a lo largo del eje X.
    Ideal para darle "latido" o vibración suave a un personaje sin romper
    su silueta.  Se puede animar con *time* y controlar la amplitud/longitud.
    """
    time       = kw.get("time", 0.0)  # segundos
    amplitude  = kw.get("amplitude", 0.15)  # desplazamiento máximo
    wavelength = kw.get("wavelength", 2.5)  # número de ondas a lo largo de X
    speed      = kw.get("speed", 1.0)       # Hz de avance de la ola

    x, y, z = vertex[:3]

    # Desplazamiento vertical en Y basado en sinusoide
    disp = amplitude * np.sin(2*np.pi*(x/wavelength + time*speed))
    y += disp

    M = kw["modelMatrix"]
    vt = M @ np.array([x, y, z, 1.0])
    return [vt[0]/vt[3], vt[1]/vt[3], vt[2]/vt[3]]


def NormalMappedPhongShader(**kw):
    """
    Phong con normal mapping (o bump si se pasa height map).
    Usa:
        - texture: albedo (opcional)
        - normalTex: normal map en espacio tangente (RGB en [0,1])
        - heightTex: height map (grises) si no hay normalTex → bump
    Parámetros opcionales:
        ambient=0.15, kd=1.0, ks=0.35, shininess=64, bumpScale=1.0
    """
    A, B, C = kw["verts"]
    u_b, v_b, w_b = kw["baryCoords"]
    u, v = kw.get("u", 0), kw.get("v", 0)
    L = np.array(kw["dirLight"], dtype=float)

    # Normal geométrica interpolada
    NA, NB, NC = np.array(A[3:6]), np.array(B[3:6]), np.array(C[3:6])
    N = u_b*NA + v_b*NB + w_b*NC
    N = N / (np.linalg.norm(N) + 1e-8)

    # Base de color (albedo)
    if (T := kw.get("texture")) is not None:
        r, g, b = T.getColor(u, v)
        albedo = np.array((r, g, b))
    else:
        albedo = np.array((1.0, 1.0, 1.0))

    # --- Tangent space (simple, sin tangentes de malla) ---
    # Construimos T y B ortogonales a N con un "up" estable
    up = np.array((0.0, 1.0, 0.0)) if abs(N[1]) < 0.9 else np.array((1.0, 0.0, 0.0))
    T = np.cross(up, N);  T /= np.linalg.norm(T) + 1e-8
    B = np.cross(N, T)

    # --- Normal map o Bump map ---
    normalTex = kw.get("normalTex", None)
    heightTex = kw.get("heightTex", None)
    if normalTex is not None:
        rn, gn, bn = normalTex.getColor(u, v)          # [0,1]
        n_t = np.array((rn*2-1, gn*2-1, bn*2-1))       # [-1,1]
    elif heightTex is not None:
        # Bump: gradiente numérico en UV
        eps = 1.0 / 1024.0
        def grayAt(uu, vv):
            rr, gg, bb = heightTex.getColor(uu, vv)
            return 0.299*rr + 0.587*gg + 0.114*bb
        h  = grayAt(u, v)
        hx = grayAt(u+eps, v) - h
        hy = grayAt(u, v+eps) - h
        scale = kw.get("bumpScale", 1.0)
        n_t = np.array((-scale*hx, -scale*hy, 1.0))
        n_t /= np.linalg.norm(n_t) + 1e-8
    else:
        n_t = np.array((0.0, 0.0, 1.0))                # sin mapa → normal original

    # Tangent → espacio "mundo/vista"
    N = T*n_t[0] + B*n_t[1] + N*n_t[2]
    N /= np.linalg.norm(N) + 1e-8

    # Iluminación Phong (Blinn)
    diff = max(0.0, N.dot(-L))
    V = np.array((0, 0, 1), dtype=float)
    H = (-L + V); H /= np.linalg.norm(H) + 1e-8
    spec = max(0.0, N.dot(H)) ** kw.get("shininess", 64)

    ambient = kw.get("ambient", 0.15)
    kd = kw.get("kd", 1.0)
    ks = kw.get("ks", 0.35)
    color = albedo * (ambient + kd*diff) + ks*spec
    color = np.clip(color, 0, 1)
    return tuple((color*255).astype(int))

def MatrixBinaryShader(verts, baryCoords, dirLight, texture, u, v, x, y, time=0, **kwargs):
    """
    Shader estilo Matrix: fondo negro con 0s y 1s verdes cayendo
    """
    import random
    import math
    
    # Parámetros configurables
    binary_size = kwargs.get('binary_size', 8)      # Tamaño de cada "celda" de binario
    speed = kwargs.get('speed', 2.0)                # Velocidad de la animación
    density = kwargs.get('density', 0.7)            # Densidad de los números (0.0 a 1.0)
    green_intensity = kwargs.get('green_intensity', 255)  # Intensidad del verde
    
    # Crear una grilla de celdas basada en las coordenadas de pantalla
    cell_x = int(x // binary_size)
    cell_y = int(y // binary_size)
    
    # Usar las coordenadas de celda para generar un "seed" pseudo-aleatorio
    # que sea consistente para cada celda pero cambie con el tiempo
    seed = cell_x * 1000 + cell_y + int(time * speed * 10)
    
    # Generar número pseudo-aleatorio basado en el seed
    pseudo_random = (seed * 9301 + 49297) % 233280
    normalized_random = (pseudo_random / 233280.0)
    
    # Decidir si mostrar un número o estar vacío
    if normalized_random > density:
        return [0, 0, 0]  # Fondo negro
    
    # Decidir si es 0 o 1
    is_one = (pseudo_random % 2) == 1
    
    # Crear efecto de "cascada" - los números aparecen y desaparecen
    cascade_offset = int(time * speed * 50) % (binary_size * 20)
    fade_factor = 1.0 - abs(((y + cascade_offset) % (binary_size * 10)) - binary_size * 5) / (binary_size * 5)
    fade_factor = max(0, fade_factor)
    
    # Aplicar iluminación básica del modelo
    A, B, C = verts
    u_bary, v_bary, w_bary = baryCoords
    
    # Interpolar normales
    nx_A, ny_A, nz_A = A[3], A[4], A[5]
    nx_B, ny_B, nz_B = B[3], B[4], B[5]
    nx_C, ny_C, nz_C = C[3], C[4], C[5]
    
    nx = u_bary * nx_A + v_bary * nx_B + w_bary * nx_C
    ny = u_bary * ny_A + v_bary * ny_B + w_bary * ny_C
    nz = u_bary * nz_A + v_bary * nz_B + w_bary * nz_C
    
    # Normalizar
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 0:
        nx, ny, nz = nx/length, ny/length, nz/length
    
    # Calcular iluminación
    light_intensity = max(0, -(nx * dirLight[0] + ny * dirLight[1] + nz * dirLight[2]))
    light_intensity = 0.3 + 0.7 * light_intensity  # Luz ambiental + direccional
    
    # Color base dependiendo de si es 0 o 1
    if is_one:
        # Verde más brillante para los 1s
        base_green = int(green_intensity * 0.9)
    else:
        # Verde más tenue para los 0s
        base_green = int(green_intensity * 0.6)
    
    # Aplicar fade y iluminación
    final_green = int(base_green * fade_factor * light_intensity)
    
    # Un toque muy sutil de verde en los otros canales para darle más "glow"
    final_red = int(final_green * 0.1)
    final_blue = int(final_green * 0.05)
    
    return [
        min(255, max(0, final_red)),
        min(255, max(0, final_green)), 
        min(255, max(0, final_blue))
    ]

def RedCameraFacingShader(verts, baryCoords, dirLight, texture, u, v, x, y, time=0, **kwargs):
    """
    Shader en tonos rojos: rojo brillante (255) si la cara mira a la cámara, 
    rojo muy oscuro (1) si mira hacia otro lado
    """
    import math
    
    # Parámetros configurables
    max_red = kwargs.get('max_red', 255)        # Rojo cuando mira a la cámara
    min_red = kwargs.get('min_red', 1)          # Rojo cuando no mira a la cámara
    smoothness = kwargs.get('smoothness', 1.0)  # Suavidad de la transición
    
    # Obtener los vértices del triángulo
    A, B, C = verts
    u_bary, v_bary, w_bary = baryCoords
    
    # Interpolar las normales en el punto actual
    nx_A, ny_A, nz_A = A[3], A[4], A[5]
    nx_B, ny_B, nz_B = B[3], B[4], B[5]
    nx_C, ny_C, nz_C = C[3], C[4], C[5]
    
    nx = u_bary * nx_A + v_bary * nx_B + w_bary * nx_C
    ny = u_bary * ny_A + v_bary * ny_B + w_bary * ny_C
    nz = u_bary * nz_A + v_bary * nz_B + w_bary * nz_C
    
    # Normalizar la normal
    normal_length = math.sqrt(nx*nx + ny*ny + nz*nz)
    if normal_length > 0:
        nx, ny, nz = nx/normal_length, ny/normal_length, nz/normal_length
    else:
        nx, ny, nz = 0, 0, 1
    
    # Vector hacia la cámara (en espacio mundial, la cámara está en Z positivo)
    # Como estamos en espacio de vista, la cámara mira hacia -Z
    camera_direction = [0, 0, 1]  # Vector hacia la cámara
    
    # Calcular qué tanto está mirando la cara hacia la cámara
    # Producto punto: 1 = mira directamente, 0 = perpendicular, -1 = mira hacia atrás
    facing_factor = nx * camera_direction[0] + ny * camera_direction[1] + nz * camera_direction[2]
    
    # Convertir de rango [-1, 1] a [0, 1] para el mapeo de color
    # Solo nos interesan las caras que miran hacia la cámara (facing_factor > 0)
    facing_factor = max(0, facing_factor)
    
    # Aplicar suavidad a la transición
    if smoothness != 1.0:
        facing_factor = pow(facing_factor, smoothness)
    
    # Mapear el factor a la intensidad de rojo
    red_intensity = int(min_red + (max_red - min_red) * facing_factor)
    
    # Crear el color final (solo rojo, otros canales en 0 o muy bajos)
    final_red = min(255, max(0, red_intensity))
    
    # Un poco de rojo en los otros canales para tonos más ricos
    final_green = int(final_red * 0.1)  # 10% del rojo en verde
    final_blue = int(final_red * 0.05)  # 5% del rojo en azul
    
    return [final_red, final_green, final_blue]

def CorruptGlitchShader(verts, baryCoords, dirLight, texture, u, v, x, y, time=0, **kwargs):
    """
    Shader de corrupción digital con glitches, separación RGB, líneas de ruido,
    y colores que cambian aleatoriamente
    """
    import math
    
    # Parámetros configurables
    intensity = kwargs.get('intensity', 0.8)        # Intensidad del glitch (0.0 a 1.0)
    rgb_separation = kwargs.get('rgb_separation', 3) # Separación de canales RGB
    noise_frequency = kwargs.get('noise_frequency', 0.1)  # Frecuencia del ruido
    glitch_speed = kwargs.get('glitch_speed', 2.0)  # Velocidad de los glitches
    
    # Generar "ruido" basado en posición y tiempo
    noise_x = int(x * noise_frequency + time * glitch_speed * 10) % 997
    noise_y = int(y * noise_frequency + time * glitch_speed * 7) % 991
    noise_t = int(time * glitch_speed * 50) % 1009
    
    # Función de ruido pseudo-aleatorio
    def noise(seed):
        n = (seed * 9301 + 49297) % 233280
        return (n / 233280.0) * 2.0 - 1.0
    
    # Diferentes tipos de glitches
    base_noise = noise(noise_x + noise_y * 1000 + noise_t)
    line_noise = noise(int(y) * 13 + noise_t)
    block_noise = noise(int(x // 8) * 17 + int(y // 8) * 23 + noise_t)
    
    # Obtener iluminación básica del modelo
    A, B, C = verts
    u_bary, v_bary, w_bary = baryCoords
    
    # Interpolar normales
    nx_A, ny_A, nz_A = A[3], A[4], A[5]
    nx_B, ny_B, nz_B = B[3], B[4], B[5] 
    nx_C, ny_C, nz_C = C[3], C[4], C[5]
    
    nx = u_bary * nx_A + v_bary * nx_B + w_bary * nx_C
    ny = u_bary * ny_A + v_bary * ny_B + w_bary * ny_C
    nz = u_bary * nz_A + v_bary * nz_B + w_bary * nz_C
    
    # Normalizar
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 0:
        nx, ny, nz = nx/length, ny/length, nz/length
    
    # Calcular iluminación básica
    light = max(0.1, -(nx * dirLight[0] + ny * dirLight[1] + nz * dirLight[2]))
    
    # === EFECTOS DE CORRUPCIÓN ===
    
    # 1. Líneas horizontales de corrupción
    if abs(line_noise) > (0.7 - intensity * 0.3):
        # Línea corrupta - colores completamente aleatorios
        r = int(abs(noise(noise_x + 100)) * 255)
        g = int(abs(noise(noise_y + 200)) * 255)
        b = int(abs(noise(noise_t + 300)) * 255)
        return [r, g, b]
    
    # 2. Bloques de corrupción 
    if abs(block_noise) > (0.8 - intensity * 0.2):
        # Bloque corrupto - colores neón intensos
        if block_noise > 0:
            return [255, 0, 255]  # Magenta
        else:
            return [0, 255, 255]  # Cyan
    
    # 3. Color base con separación RGB
    base_intensity = light * 180
    
    # Separar canales RGB con offsets diferentes
    r_offset = int(rgb_separation * noise(noise_t + 50))
    g_offset = int(rgb_separation * noise(noise_t + 150))
    b_offset = int(rgb_separation * noise(noise_t + 250))
    
    # Samplear cada canal desde posiciones ligeramente diferentes
    r_noise = noise((x + r_offset) * noise_frequency + noise_y)
    g_noise = noise((x + g_offset) * noise_frequency + noise_y + 500)
    b_noise = noise((x + b_offset) * noise_frequency + noise_y + 1000)
    
    # 4. Aplicar distorsión por canal
    r_intensity = base_intensity + r_noise * intensity * 100
    g_intensity = base_intensity + g_noise * intensity * 100  
    b_intensity = base_intensity + b_noise * intensity * 100
    
    # 5. Efectos especiales aleatorios
    special_glitch = noise(noise_x * 3 + noise_y * 7 + noise_t * 2)
    
    if special_glitch > (0.95 - intensity * 0.1):
        # Píxeles completamente corruptos - colores imposibles
        return [255, 255, 0]  # Amarillo brillante
    elif special_glitch < (-0.95 + intensity * 0.1):
        # Píxeles "muertos"
        return [0, 0, 0]
    
    # 6. Inversión de colores aleatoria
    invert_noise = noise(int(x // 4) + int(y // 4) * 100 + noise_t)
    if abs(invert_noise) > (0.85 - intensity * 0.2):
        r_intensity = 255 - r_intensity
        g_intensity = 255 - g_intensity
        b_intensity = 255 - b_intensity
    
    # 7. Aplicar "bit crushing" - reducir precisión de color
    if intensity > 0.5:
        bit_reduction = int((intensity - 0.5) * 8)
        mask = 255 ^ ((1 << bit_reduction) - 1)
        r_intensity = int(r_intensity) & mask
        g_intensity = int(g_intensity) & mask  
        b_intensity = int(b_intensity) & mask
    
    # Clamp final
    final_r = min(255, max(0, int(r_intensity)))
    final_g = min(255, max(0, int(g_intensity)))
    final_b = min(255, max(0, int(b_intensity)))
    
    return [final_r, final_g, final_b]