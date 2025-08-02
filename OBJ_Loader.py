def LoadObj(filename):
    vertices = []
    uvs = []
    normals = []
    faces = []
    mtl_map = {}
    mtl = None
    mtllib = None

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("mtllib"):
                mtllib = line.strip().split()[1]
            elif line.startswith("usemtl"):
                mtl = line.strip().split()[1]
            elif line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.strip().split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("vn "):
                parts = line.strip().split()
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    vals = part.split("/")
                    vi = int(vals[0]) - 1
                    ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] != '' else None
                    ni = int(vals[2]) - 1 if len(vals) > 2 and vals[2] != '' else None
                    face.append((vi, ti, ni))
                faces.append((face, mtl))

    if mtllib:
        with open("models/" + mtllib, "r") as f:
            curr_mtl = None
            for line in f:
                if line.strip().startswith("newmtl"):
                    curr_mtl = line.strip().split()[1]
                elif line.strip().startswith("map_Kd") and curr_mtl:
                    mtl_map[curr_mtl] = line.strip().split()[1]
    return vertices, uvs, normals, faces, mtl_map