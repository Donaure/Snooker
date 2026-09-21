"""Align the licensed cue shaft mesh and join it to the 9.5 mm ferrule.

Preserves the source UVs, removes its old ferrule/tip, and refines the original
nine-sided shaft surface. Only a zero-mass, non-colliding visual asset changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets/cue/sketchfab_pool_table_traditional'


def build() -> None:
    lines = (ASSETS / 'pool_cue_visual_local.obj').read_text().splitlines()
    vertices = np.array([[float(x) for x in line.split()[1:4]]
                         for line in lines if line.startswith('v ')], dtype=np.float64)
    uv = np.array([[float(x) for x in line.split()[1:3]]
                   for line in lines if line.startswith('vt ')], dtype=np.float64)
    # This source asset's final wooden ring precedes x=0.706. All subsequent
    # rings belong to the old white ferrule and tip, which must not remain.
    wood = vertices[:, 0] < .706
    order = np.argsort(vertices[wood, 0])
    rings = np.split(vertices[wood][order],
                     np.flatnonzero(np.diff(vertices[wood][order, 0]) > .004) + 1)
    centers, radii, axial = [], [], []
    for ring in rings:
        yz = ring[:, 1:]
        fit = np.linalg.lstsq(np.column_stack((2*yz, np.ones(len(yz)))),
                              np.sum(yz*yz, axis=1), rcond=None)[0]
        centers.append(fit[:2])
        radii.append(np.sqrt(fit[2] + fit[:2] @ fit[:2]))
        axial.append(float(np.mean(ring[:, 0])))
    centers = np.array(centers)
    end = axial[-1]
    profile_x = [-.725, -.30, .10, .40, .60, .710]
    profile_r = [.0145, .011, .008, .006, .00475, .00475]

    def transform(point: np.ndarray, surface: bool) -> np.ndarray:
        x = float(point[0])
        center = np.array([np.interp(x, axial, centers[:, j]) for j in range(2)])
        radial = point[1:] - center
        # Flatten the last ring and terminate exactly at the new ferrule.
        new_x = .710 if x > .705 else -.725 + (x + .725) * (1.435 / (end + .725))
        radius = float(np.interp(new_x, profile_x, profile_r))
        denominator = np.linalg.norm(radial) if surface else np.interp(x, axial, radii)
        return np.r_[new_x, radial * radius / max(float(denominator), 1e-12)]

    output_vertices, output_uv, faces = [], [], []
    indices: dict[tuple[float, ...], int] = {}

    def emit(point: np.ndarray, texcoord: np.ndarray, surface: bool) -> int:
        vertex = transform(point, surface)
        key = tuple(np.round(np.r_[vertex, texcoord], 10))
        if key not in indices:
            indices[key] = len(output_vertices) + 1
            output_vertices.append(vertex)
            output_uv.append(texcoord)
        return indices[key]

    for line in lines:
        if not line.startswith('f '):
            continue
        corners = [[int(i)-1 for i in token.split('/')[:2]] for token in line.split()[1:]]
        if not all(wood[c[0]] for c in corners):
            continue
        for k in range(1, len(corners)-1):
            triangle = [corners[0], corners[k], corners[k+1]]
            points = vertices[[c[0] for c in triangle]]
            texcoords = uv[[c[1] for c in triangle]]
            surface = np.ptp(points[:, 0]) > 1e-4
            grid = {}
            n = 4
            for i in range(n+1):
                for j in range(n+1-i):
                    weights = np.array([n-i-j, i, j], dtype=np.float64) / n
                    grid[i, j] = emit(weights @ points, weights @ texcoords, surface)
            for i in range(n):
                for j in range(n-i):
                    faces.append((grid[i, j], grid[i+1, j], grid[i, j+1]))
                    if i+j < n-1:
                        faces.append((grid[i+1, j], grid[i+1, j+1], grid[i, j+1]))

    result = ['# Derived from pool_cue_visual_local.obj; same source license and UVs.',
              '# Rebuild with scripts/assets/build_snooker_cue_visual.py.']
    result += ['v ' + ' '.join(f'{x:.10f}' for x in p) for p in output_vertices]
    result += ['vt ' + ' '.join(f'{x:.10f}' for x in p) for p in output_uv]
    # The source extraction mirrored local X without reversing triangle winding.
    # Reverse it here so the visible surface faces outward rather than showing
    # the inside wall and the back of the ferrule from oblique viewpoints.
    result += ['f ' + ' '.join(f'{i}/{i}' for i in reversed(face)) for face in faces]
    path = ASSETS / 'snooker_cue_shaft_visual.obj'
    path.write_text('\n'.join(result) + '\n')
    print(f'wrote={path} vertices={len(output_vertices)} triangles={len(faces)}')


if __name__ == '__main__':
    build()
