class Renderer(object):
    def __init__(self, screen):
        self.screen = screen
        _, _, self.width, self.height = self.screen.get_rect()

        self.glColor(1, 1, 1)
        self.glLimpiarColor(0, 0, 0)

        self.glLimpiar()

    def glLimpiarColor(self, r, g, b):
        r = min(1, max(0, r))
        g = min(1, max(0, g))
        b = min(1, max(0, b))

        self.clearColor = [r, g, b]

    def glColor(self, r, g, b):
        r = min(1, max(0, r))
        g = min(1, max(0, g))
        b = min(1, max(0, b))

        self.currColor = [r, g, b]
    
    def glLimpiar(self):
        color = [int(i * 255) for i in self.clearColor]
        self.screen.fill(color)

        self.frameBuffer = [[self.clearColor for y in range(self.height)] for x in range(self.width)]
        
    def glPunto(self, x, y, color = None):
        x = round(x)
        y = round(y)

        if (0 <= x < self.width) and (0 <= y < self.height):
            color = [int(i * 255) for i in (color or self.currColor)]

            self.screen.set_at((x, self.height - 1 - y), color)
            self.frameBuffer[x][y] = color
    
    def glLinea(self, p0, p1, color = None):
        x0 = p0[0]
        x1 = p1[0]
        y0 = p0[1]
        y1 = p1[1]

        if x0 == x1 and y0 == y1:
            self.glPunto(x0, y0)
            return

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        offset = 0
        limit = 0.75
        m = dy / dx
        y = y0

        for x in range(round(x0), round(x1) + 1):
            if steep:
                self.glPunto(y, x, color or self.currColor)
            else:
                self.glPunto(x, y, color or self.currColor)

            offset += m

            if offset >= limit:
                if y0 < y1:
                    y += 1
                else:
                    y -= 1

                limit += 1

    def glPoligono(self, vertices, color=None):
        for i in range(len(vertices)):
            p0 = vertices[i]
            p1 = vertices[(i+1) % len(vertices)]
            self.glLinea(p0, p1, color)

    def glLlenarPoligono(self, vertices, color=None):
        llenar_color = color or self.currColor

        ys = [v[1] for v in vertices]
        y_min = int(min(ys))
        y_max = int(max(ys))

        for y in range(y_min, y_max + 1):
            intersecciones = []

            for i in range(len(vertices)):
                x0, y0 = vertices[i]
                x1, y1 = vertices[(i+1) % len(vertices)]

                if y0 == y1:
                    continue

                if (y >= min(y0, y1)) and (y < max(y0, y1)):
                    t = (y - y0) / (y1 - y0)
                    x = x0 + t * (x1 - x0)
                    intersecciones.append(x)

            if not intersecciones:
                continue

            intersecciones.sort()

            for i in range(0, len(intersecciones), 2):
                x_start = int(round(intersecciones[i]))

                if i+1 < len(intersecciones):
                    x_end = int(round(intersecciones[i+1]))

                    for x in range(x_start, x_end + 1):
                        self.glPunto(x, y, llenar_color)