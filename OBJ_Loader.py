import os  
  
def LoadObj(filename):
    vertices, uvs, normals, faces = [], [], [], []
    mtl_map = {}
    mtl = None
    mtllib = None

    obj_dir = os.path.dirname(filename)          # ← NUEVO

    # --- 1.  Leer el .obj -------------------------------------------------
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("mtllib"):
                mtllib = line.strip().split()[1]
            elif line.startswith("usemtl"):
                mtl = line.strip().split()[1]
            elif line.startswith("v "):
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("vt "):
                parts = line.strip().split()
                u, v = float(parts[1]), float(parts[2])
                uvs.append([float(u), float(v)])
            elif line.startswith("vn "):
                _, nx, ny, nz = line.strip().split()
                normals.append([float(nx), float(ny), float(nz)])
            
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    vi, *rest = part.split("/")
                    ti = int(rest[0])-1 if len(rest) > 0 and rest[0] else None
                    ni = int(rest[1])-1 if len(rest) > 1 and rest[1] else None
                    face.append((int(vi)-1, ti, ni))
                
                # Triangular quads: si tiene 4 vértices, crear 2 triángulos
                if len(face) == 4:
                    faces.append(([face[0], face[1], face[2]], mtl))  # Triángulo 1
                    faces.append(([face[0], face[2], face[3]], mtl))  # Triángulo 2
                else:
                    faces.append((face, mtl))  # Ya es triángulo o otro formato

    # --- 2.  Leer el .mtl (si existe) ------------------------------------
    if mtllib:
        with open(os.path.join(obj_dir, mtllib), "r") as f:
            curr_mtl = None
            for line in f:
                if line.startswith("newmtl"):
                    curr_mtl = line.strip().split()[1]
                elif line.startswith("map_Kd") and curr_mtl:
                    tex_file = line.strip().split()[1]
                    # guarda la ruta completa de la textura
                    mtl_map[curr_mtl] = os.path.join(obj_dir, tex_file)

    return vertices, uvs, normals, faces, mtl_map