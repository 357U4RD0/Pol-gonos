import os, pygame, numpy as np
from collections import defaultdict
from GraphicLibrary import Renderer
from OBJ_Loader import LoadObj
from Model import Model
from MathLibrary import LookAt
from Shaders import VertexShader
from Texture import Texture
from BMP_Writer import GenerateBmp

from custom_shaders import (
    RedCameraFacingShader, HalftoneToonShader, 
    CorruptGlitchShader, WaveVertexShader, 
    MatrixBinaryShader,
)

# Rutas 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BG_PATH    = os.path.join(BASE_DIR, "ImagenFondo.jpg")
SHOOTS_DIR = os.path.join(BASE_DIR, "Fotos")

# Ventana / renderer
width, height = 1280, 720
pygame.init()
screen   = pygame.display.set_mode((width, height), pygame.SCALED)
clock    = pygame.time.Clock()
renderer = Renderer(screen)
renderer.primitiveType = 2
renderer.dirLight = np.array((0.5, 0.7, -1.0))

bg_surf = None
if os.path.exists(BG_PATH):
    bg_surf = pygame.image.load(BG_PATH).convert()
    bg_surf = pygame.transform.smoothscale(bg_surf, (width, height))

def paint_background():
    if bg_surf is None:
        return
    screen.blit(bg_surf, (0, 0))
    arr = pygame.surfarray.array3d(bg_surf)  # (W,H,3)
    for x in range(width):
        col = arr[x]
        for y_scr in range(height):
            y_fb = height - 1 - y_scr
            r, g, b = map(int, col[y_scr])
            renderer.frameBuffer[x][y_fb] = [r, g, b]

def build_models_from_obj(obj_path, v_sh, f_sh,
                          translation=(0,0,-6), rotation_deg=(0,0,0), scale=(1,1,1),
                          frag_params=None, vert_params=None,
                          normalize=True, target_radius=1.2):

    models = []
    vertices, uvs, normals, faces, mtl_map = LoadObj(obj_path)
    textures = {mat: Texture(tex_path) for mat, tex_path in mtl_map.items()}

    faces_by_mat = defaultdict(list)
    for tri, mat in faces:
        faces_by_mat[mat].append(tri)

    all_xyz = np.array([vertices[vi]
                        for tris in faces_by_mat.values()
                        for tri in tris
                        for (vi, _, _) in tri], dtype=float)
    if len(all_xyz) == 0:
        return models

    centroid = all_xyz.mean(axis=0)
    radius   = np.max(np.linalg.norm(all_xyz - centroid, axis=1))
    unit_mul = (target_radius / max(radius, 1e-6)) if normalize else 1.0
    scale_mul = np.array(scale, dtype=float) * unit_mul

    for mat, tris in faces_by_mat.items():
        buf = []
        for tri in tris:
            for vi, ti, ni in tri:
                x, y, z = vertices[vi] - centroid      # center
                x, y, z = x*scale_mul[0], y*scale_mul[1], z*scale_mul[2]
                nx, ny, nz = normals[ni] if ni is not None else (0, 0, 1)
                u, v = uvs[ti] if ti is not None else (0, 0)
                buf += [x, y, z, nx, ny, nz, u, v]

        m = Model()
        m.vertices        = buf
        m.vertexShader    = v_sh
        m.fragmentShader  = f_sh
        m.texture         = textures.get(mat)

        m.translation     = list(translation)
        rx, ry, rz        = rotation_deg
        m.rotation        = [np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)]

        if frag_params: m.fragmentParams = frag_params.copy()
        if vert_params: m.vertexParams   = vert_params.copy()
        models.append(m)
    return models

SCENE = [
    (os.path.join(BASE_DIR,"modelos","14082_WWII_Plane_Japan_Kawasaki_Ki-61_v1_L2.obj"),
     (2.6,2.6,2.6), (-10.8, 6.50, -9.2), (-1,  57, 30),
     (VertexShader, MatrixBinaryShader),
     {"frag":{"speed":0.8,"scale":60,"palette":"rainbow"}}),

    (os.path.join(BASE_DIR,"modelos","10621_CoastGuardHelicopter.obj"),
     (1,1,1), ( 3.2, 3.15, 0.3), (110, 200, 0),
     (VertexShader, RedCameraFacingShader),
     {"frag":{"rimIntensity":1.2}}),
    
    (os.path.join(BASE_DIR,"modelos","Cyprys_House.obj"),
     (6,6,6), (-6, -2.90, -6.8), (0, 20, 0),
     (VertexShader, HalftoneToonShader),
     {"frag":{"levels":4,"ambient":0.18,"dotSize":4}}),

    (os.path.join(BASE_DIR,"modelos","11717_bicycle_v2_L1.obj"),
     (1.5,1.5,1.5), ( 8.6, -3.90, -7.5), (180, -10, 90),
     (WaveVertexShader, CorruptGlitchShader),
     {"frag":{"intensity":1.15},
      "vert":{"amplitude":0.12,"wavelength":2.8,"speed":0.7}}),
]

renderer.models = []
for (obj_path, scale, trans, rot, (v_sh, f_sh), params) in SCENE:
    models = build_models_from_obj(
        obj_path, v_sh, f_sh,
        translation=trans, rotation_deg=rot, scale=scale,
        frag_params=params.get("frag", {}), vert_params=params.get("vert", {})
    )
    renderer.models.extend(models)
for i, model in enumerate(renderer.models):
    print(f"Modelo {i}: {len(model.vertices)} elementos totales, {len(model.vertices)//8} vértices")

renderer.viewMatrix = LookAt(
    np.array((0.0, 1.15, 6.1)),   
    np.array((0.0,-0.05,-8.0)),   
    np.array((0.0, 1.0,  0.0))
)
renderer.dirLight = np.array((-0.25, 0.9, -0.35))

os.makedirs(SHOOTS_DIR, exist_ok=True)
saved = False
running = True
while running:
    _ = clock.tick(60)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    renderer.glClear()
    paint_background()
    renderer.glRender()
    pygame.display.flip()

    if not saved:
        out = os.path.join(SHOOTS_DIR, "Scene.bmp")
        GenerateBmp(out, width, height, 3, renderer.frameBuffer)
        print(f"✓ guardado {out}")
        saved = True

pygame.quit()