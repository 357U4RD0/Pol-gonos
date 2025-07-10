import pygame
from gl import Renderer
from BMP_Writer import GenerateBMP

width = 1920//2
height = 1080//2

screen = pygame.display.set_mode((width, height), pygame.SCALED)
reloj = pygame.time.Clock()

rend = Renderer(screen)

rend.glLimpiarColor(0, 0, 0)

polygons = [
    #Coordenadas para todos los polígonos

    # Polígono 1
    [(165, 380), (185, 360), (180, 330), (207, 345),(233, 330), (230, 360), (250, 380), (220, 385), (205, 410), (193, 383)],
    # Polígono 2
    [(321, 335), (288, 286), (339, 251), (374, 302)],
    # Polígono 3
    [(377, 249), (411, 197), (436, 249)],
    # Polígono 4
    [(413, 177), (448, 159), (502, 88),  (553, 53), (535, 36),  (676, 37),  (660, 52),  (750, 145), (761, 179), (672, 192), (659, 214), (615, 214), (632, 230), (580, 230), (597, 215), (552, 214), (517, 144), (466, 180)],
]

# Polígono 5
hoyo = [
    (682, 175), (708, 120),
    (735, 148), (739, 170),
]

colors = [
    (0.5, 1, 0.5),
    (1, 0.55, 0),
    (1, 0.4, 0.4),
    (1, 0.84, 0),
]

isRunning = True
while isRunning:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
    
    rend.glLimpiar()

    for i, poly in enumerate(polygons):
        rend.glColor(*colors[i])
        rend.glLlenarPoligono(poly)
        rend.glColor(1, 1, 1)
        rend.glPoligono(poly)

    rend.glLlenarPoligono(hoyo, rend.clearColor)
    rend.glColor(1, 1, 1)
    rend.glPoligono(hoyo)

    pygame.display.flip()
    reloj.tick(60)

GenerateBMP("output.bmp", width, height, 3, rend.frameBuffer)
pygame.quit()