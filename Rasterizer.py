import os
import pygame
import numpy as np
from collections import defaultdict
from GraphicLibrary import Renderer
from OBJ_Loader import LoadObj
from model import Model
from MathLibrary import LookAt, RotationMatrix
from shaders import VertexShader, GouraudShader
from BMP_Writer import GenerateBmp

width, height = 720, 480
pygame.init()
screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()
renderer = Renderer(screen)
renderer.primitiveType = 2

vertices, uvs, normals, faces, mtl_map = LoadObj("models/Barco.obj")

faces_by_mat = defaultdict(list)
for tri, mat in faces:
    faces_by_mat[mat].append(tri)

all_verts_xyz = []
for mat, tris in faces_by_mat.items():
    for tri in tris:
        for vi, ti, ni in tri:
            x, y, z = vertices[vi]
            all_verts_xyz.append([x, y, z])
all_verts_xyz = np.array(all_verts_xyz)
centroid = all_verts_xyz.mean(axis=0)

for mat, tris in faces_by_mat.items():
    model = Model()
    buf = []
    for tri in tris:
        for vi, ti, ni in tri:
            x, y, z = vertices[vi]
            nx, ny, nz = normals[ni] if ni is not None else (0, 0, 1)
            u, v = uvs[ti] if ti is not None else (0, 0)
            x -= centroid[0]
            y -= centroid[1]
            z -= centroid[2]
            buf += [x, y, z, nx, ny, nz, u, v]
    model.vertices = buf
    model.translation = [0, 0, 0]
    model.scale = [6, 6, 6]
    model.vertexShader = VertexShader
    model.fragmentShader = GouraudShader
    renderer.models.append(model)

running = True
while running:
    dt = clock.tick(60) / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                renderer.primitiveType = 0
            elif event.key == pygame.K_2:
                renderer.primitiveType = 1
            elif event.key == pygame.K_3:
                renderer.primitiveType = 2
    renderer.glClear()
    renderer.glRender()
    pygame.display.flip()

pygame.quit()