def cargar_modelo(path):
    verts = []          
    model_vertices = [] 

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "v":
                if len(parts) < 4:
                    print(f"Warning: vértice malformado: {line.strip()}")
                    continue
                x, y, z = map(float, parts[1:4])
                verts.append((x, y, z))

            elif parts[0] == "f":
                idx = []
                for tok in parts[1:]:
                    token_v = tok.split("/")[0]
                    try:
                        raw = int(token_v)
                    except ValueError:
                        print(f"Warning: no pude parsear índice '{tok}'")
                        idx = []
                        break

                    if raw > 0:
                        vi = raw - 1
                    elif raw < 0:
                        vi = len(verts) + raw
                    else:
                        idx = []
                        break

                    if vi < 0 or vi >= len(verts):
                        print(f"Warning: índice {raw} (=>{vi}) fuera de rango (0–{len(verts)-1})")
                        idx = []
                        break

                    idx.append(vi)

                if len(idx) < 3:
                    continue

                for i in range(1, len(idx) - 1):
                    for j in (0, i, i + 1):
                        x, y, z = verts[idx[j]]
                        model_vertices.extend([x, y, z])

    return model_vertices