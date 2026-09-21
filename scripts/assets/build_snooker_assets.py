"""Build a detailed original snooker table with real slate openings and rubber jaws.

All geometry and materials are procedural, without downloaded artwork. Contact
parameters and collision profiles are shared with the calibration fixtures.
"""
from __future__ import annotations

import math
from xml.etree.ElementTree import Element, SubElement as E, indent, tostring

import numpy as np

from _bootstrap import add_src_to_path
ROOT = add_src_to_path()
from snooker_env.snooker_geometry import (
    BALL_RADIUS, BAULK_Y, D_RADIUS, FLOOR_Z, NAMES, SPOTS, SURFACE_Z,
)
from snooker_env.snooker_parameters import SNOOKER_PHYSICS
from snooker_env.snooker_table_assets import (
    CORNER_RUNOUT, CUSHION_PROFILE, HALF_LENGTH, HALF_WIDTH, MIDDLE_RUNOUT,
    POCKETS, SLATE_THICKNESS, cloth_polygons, extrusion_faces, jaw_sections,
    profile_vertices,
)


def numbers(values) -> str:
    return ' '.join(f'{v:.9g}' for v in np.asarray(values).ravel())


class Builder:
    def __init__(self, asset, world):
        self.asset, self.world = asset, world

    def mesh(self, name, vertices, faces, material, *, collision=None, shell=False):
        E(self.asset, 'mesh', name=name, vertex=numbers(vertices),
          face=' '.join(str(v) for face in faces for v in face),
          **({'inertia': 'shell'} if shell else {}))
        attrs = dict(name=name, type='mesh', mesh=name, material=material)
        if collision:
            attrs['class'] = collision
        else:
            attrs.update(contype='0', conaffinity='0')
        return E(self.world, 'geom', **attrs)

    def box(self, name, position, size, material, *, collision=None):
        attrs = dict(name=name, type='box', pos=numbers(position), size=numbers(size), material=material)
        if collision:
            attrs['class'] = collision
        else:
            attrs.update(contype='0', conaffinity='0')
        return E(self.world, 'geom', **attrs)

    def lines(self, name, segments, radius, material, sides=6):
        vertices, faces = [], []
        for a, b in segments:
            a, b = np.array(a), np.array(b)
            axis = b-a
            axis /= np.linalg.norm(axis)
            ref = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([1., 0., 0.])
            one = np.cross(axis, ref)
            one /= np.linalg.norm(one)
            two = np.cross(axis, one)
            start = len(vertices)
            for p in (a, b):
                vertices.extend(p + radius*(one*math.cos(t)+two*math.sin(t))
                                for t in np.linspace(0, 2*math.pi, sides, endpoint=False))
            faces.extend(tuple(start+i for i in face) for face in extrusion_faces(sides))
        return self.mesh(name, vertices, faces, material)

    def lathe(self, name, center, profile, material, count=48):
        vertices = [(center[0]+r*math.cos(t), center[1]+r*math.sin(t), z)
                    for r, z in profile for t in np.linspace(0, 2*math.pi, count, endpoint=False)]
        faces = []
        for k in range(len(profile)-1):
            for i in range(count):
                a, b = k*count+i, k*count+(i+1)%count
                faces += [(a, b, b+count), (a, b+count, a+count)]
        faces += [(0, i+1, i) for i in range(1, count-1)]
        base = (len(profile)-1)*count
        faces += [(base, base+i, base+i+1) for i in range(1, count-1)]
        return self.mesh(name, vertices, faces, material)


def add_materials(asset):
    E(asset, 'texture', name='snooker_cloth_weave', type='2d', builtin='flat',
      rgb1='.075 .352 .181', rgb2='.080 .360 .189', mark='random', markrgb='.072 .342 .176', random='.17',
      width='128', height='128')
    E(asset, 'material', name='mat-surface', texture='snooker_cloth_weave',
      texrepeat='18 36', rgba='1 1 1 1', specular='.12', shininess='.08')
    E(asset, 'texture', name='snooker_bed_cloth_weave', type='2d', builtin='flat',
      rgb1='.084 .394 .203', rgb2='.090 .403 .212', mark='random', markrgb='.081 .383 .197', random='.17',
      width='128', height='128')
    E(asset, 'material', name='snooker_bed_cloth', texture='snooker_bed_cloth_weave',
      texrepeat='18 36', rgba='1 1 1 1', specular='.12', shininess='.08')
    materials = {
        'snooker_wood': ('.17 .070 .031 1', '.28', '.42'),
        'snooker_panel': ('.095 .029 .014 1', '.20', '.3'),
        'snooker_moulding': ('.19 .077 .031 1', '.30', '.5'),
        'snooker_wood_grain': ('.080 .026 .009 1', '.18', '.3'),
        'snooker_leather': ('.23 .112 .049 1', '.10', '.08'),
        'snooker_stitch': ('.43 .30 .157 1', '.05', '.03'),
        'snooker_brass': ('.40 .28 .091 1', '.55', '.7'),
        'snooker_iron': ('.077 .074 .063 1', '.40', '.65'),
        'snooker_net': ('.35 .31 .24 1', '.04', '.01'),
        'snooker_shadow': ('.018 .013 .009 1', '.02', '.01'),
        'snooker_slate': ('.047 .050 .049 1', '.06', '.03'),
        'snooker_marking': ('.50 .54 .42 1', '.02', '.01'),
        'snooker_ball_finish': ('1 1 1 1', '.40', '.74'),
    }
    for name, (rgba, specular, shininess) in materials.items():
        E(asset, 'material', name=name, rgba=rgba, specular=specular, shininess=shininess)


def add_slate(b: Builder):
    polygons = cloth_polygons(24)
    center = polygons[0]
    contact = b.box('snooker_cloth_center', (0, 0, SURFACE_Z-SLATE_THICKNESS/2),
                    (center[1][0], center[2][1], SLATE_THICKNESS/2),
                    'mat-surface', collision='snooker_cloth')
    contact.set('rgba', '0 0 0 0')
    cloth_vertices, slate_vertices, faces = [], [], []
    for i, polygon in enumerate(polygons):
        # Duplicate corner vertices can occur at a circle extremum; drop them.
        unique = []
        for point in polygon:
            if not unique or np.linalg.norm(np.array(point)-unique[-1]) > 1e-9:
                unique.append(point)
        if len(unique) > 1 and np.linalg.norm(np.array(unique[0])-unique[-1]) < 1e-9:
            unique.pop()
        if len(unique) < 3:
            continue
        prism_faces = extrusion_faces(len(unique))
        vertices = [(x, y, z) for z in (SURFACE_Z-SLATE_THICKNESS, SURFACE_Z) for x, y in unique]
        if i:
            contact = b.mesh(f'snooker_cloth_shelf_{i-1}', vertices, prism_faces,
                             'mat-surface', collision='snooker_cloth')
            contact.set('rgba', '0 0 0 0')
        # Separate visual cloth and slate surfaces retain exactly the same
        # support footprint and top elevation as the collision geometry.
        # A full-depth green sidewall would resemble an exposed block of felt.
        offset = len(cloth_vertices)
        cloth_vertices.extend((x, y, z) for z in (SURFACE_Z-.002, SURFACE_Z) for x, y in unique)
        slate_vertices.extend((x, y, z) for z in (SURFACE_Z-SLATE_THICKNESS, SURFACE_Z-.002)
                              for x, y in unique)
        faces.extend(tuple(offset+j for j in face) for face in prism_faces)
    b.mesh('snooker_bed_cloth_visual', cloth_vertices, faces, 'snooker_bed_cloth')
    b.mesh('snooker_bed_slate_visual', slate_vertices, faces, 'snooker_slate')


def add_cushions(b: Builder):
    def closed_mesh(name, vertices, faces, *, collision=None):
        # Reflection and reversed extrusion directions change handedness.
        # Orient each single closed cushion solid before MuJoCo derives its
        # render normals; vertices and convex collision hull stay unchanged.
        points = np.asarray(vertices, dtype=np.float64)
        points = points - points.mean(axis=0)
        triangles = points[np.asarray(faces, dtype=np.int64)]
        volume = np.einsum('ij,ij->i', triangles[:, 0],
                           np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        if abs(volume) < 1e-14:
            raise ValueError(f'Degenerate closed cushion mesh: {name}')
        if volume < 0:
            faces = [(first, third, second) for first, second, third in faces]
        return b.mesh(name, vertices, faces, 'mat-surface', collision=collision)

    for side in (-1, 1):
        for half in (-1, 1):
            start = (side*HALF_WIDTH, half*MIDDLE_RUNOUT)
            end = (side*HALF_WIDTH, half*(HALF_LENGTH-CORNER_RUNOUT))
            name = f'snooker_cushion_side_{side}_{half}'
            closed_mesh(name, profile_vertices(start, end, (side, 0)),
                        extrusion_faces(len(CUSHION_PROFILE)), collision='snooker_cushion')
        start = (-HALF_WIDTH+CORNER_RUNOUT, side*HALF_LENGTH)
        end = (HALF_WIDTH-CORNER_RUNOUT, side*HALF_LENGTH)
        closed_mesh(f'snooker_cushion_end_{side}', profile_vertices(start, end, (0, side)),
                    extrusion_faces(len(CUSHION_PROFILE)), collision='snooker_cushion')
    for name, rings in jaw_sections():
        count=len(CUSHION_PROFILE)
        for i,(a,z) in enumerate(zip(rings,rings[1:])):
            geom=closed_mesh(f'snooker_cushion_jaw_{name}_{i}',a+z,extrusion_faces(count),
                             collision='snooker_jaw')
            geom.set('rgba','0 0 0 0')
            geom.set('group','3')
        vertices=[point for ring in rings for point in ring]
        faces=[]
        for k in range(len(rings)-1):
            for i in range(count):
                a,c=k*count+i,k*count+(i+1)%count
                faces += [(a,c,c+count),(a,c+count,a+count)]
        faces += [(0,i+1,i) for i in range(1,count-1)]
        base=(len(rings)-1)*count
        faces += [(base,base+i,base+i+1) for i in range(1,count-1)]
        closed_mesh(f'snooker_jaw_visual_{name}',vertices,faces)


def add_cabinet(b: Builder):
    # Six separate rail tops and apron panels leave actual openings at pockets.
    for sx in (-1, 1):
        for sy in (-1, 1):
            y0, y1 = .074, HALF_LENGTH-.030
            cy, hy = sy*(y0+y1)/2, (y1-y0)/2
            label = f'side_{sx}_{sy}'
            b.box(f'snooker_rail_{label}', (sx*(HALF_WIDTH+.103), cy, SURFACE_Z+.010),
                  (.048, hy, .031), 'snooker_wood')
            b.box(f'snooker_rail_outer_bead_{label}', (sx*(HALF_WIDTH+.151), cy, SURFACE_Z+.019),
                  (.006, hy, .013), 'snooker_moulding')
            b.box(f'snooker_apron_{label}', (sx*(HALF_WIDTH+.075), cy, SURFACE_Z-.133),
                  (.057, hy, .114), 'snooker_wood')
            b.box(f'snooker_apron_panel_{label}', (sx*(HALF_WIDTH+.133), cy, SURFACE_Z-.135),
                  (.002, hy-.045, .072), 'snooker_panel')
            for dz in (-.052, -.222):
                b.box(f'snooker_apron_bead_{label}_{dz}', (sx*(HALF_WIDTH+.138), cy, SURFACE_Z+dz),
                      (.009, hy+.008, .006), 'snooker_moulding')
            # Subtle inlaid lines on the polished rail, visible at close range.
            b.lines(f'snooker_rail_inlay_{label}',
                    [((sx*(HALF_WIDTH+u), sy*y0, SURFACE_Z+.0411),
                      (sx*(HALF_WIDTH+u), sy*y1, SURFACE_Z+.0411)) for u in (.066, .142)],
                    .00065, 'snooker_brass')
            grain = []
            for k in range(9):
                u = .072+k*.007
                points = [(sx*(HALF_WIDTH+u+.0012*math.sin(t*9+k)), sy*t, SURFACE_Z+.04115)
                          for t in np.linspace(y0+.008, y1-.008, 48)]
                grain.extend(zip(points, points[1:]))
            b.lines(f'snooker_rail_grain_{label}', grain, .00018, 'snooker_wood_grain', sides=4)
        end_width = HALF_WIDTH-.030
        b.box(f'snooker_rail_end_{sx}', (0, sx*(HALF_LENGTH+.102), SURFACE_Z+.010),
              (end_width, .048, .031), 'snooker_wood')
        b.box(f'snooker_rail_end_bead_{sx}', (0, sx*(HALF_LENGTH+.15), SURFACE_Z+.019),
              (end_width, .006, .013), 'snooker_moulding')
        b.box(f'snooker_apron_end_{sx}', (0, sx*(HALF_LENGTH+.075), SURFACE_Z-.133),
              (end_width, .057, .114), 'snooker_wood')
        b.box(f'snooker_apron_end_panel_{sx}', (0, sx*(HALF_LENGTH+.133), SURFACE_Z-.135),
              (end_width-.045, .002, .072), 'snooker_panel')
        for dz in (-.052, -.222):
            b.box(f'snooker_apron_end_bead_{sx}_{dz}', (0, sx*(HALF_LENGTH+.138), SURFACE_Z+dz),
                  (end_width+.008, .009, .006), 'snooker_moulding')
        b.lines(f'snooker_rail_end_inlay_{sx}',
                [((-end_width, sx*(HALF_LENGTH+u), SURFACE_Z+.0411),
                  (end_width, sx*(HALF_LENGTH+u), SURFACE_Z+.0411)) for u in (.066, .141)],
                .00065, 'snooker_brass')
    b.box('snooker_slate_bed_support', (0, 0, SURFACE_Z-.078),
          (HALF_WIDTH-.045, HALF_LENGTH-.05, .030), 'snooker_shadow')
    # Eight turned legs, with collars, taper and levelling feet.
    for sx in (-1, 1):
        for i, y in enumerate((-1.52, -.52, .52, 1.52)):
            x = sx*.75
            top = SURFACE_Z-.18
            profile = [(.071,FLOOR_Z+.013),(.083,FLOOR_Z+.03),(.083,FLOOR_Z+.061),
                       (.059,FLOOR_Z+.077),(.048,FLOOR_Z+.13),(.063,FLOOR_Z+.23),
                       (.085,FLOOR_Z+.35),(.088,FLOOR_Z+.395),(.067,FLOOR_Z+.423),
                       (.067,top-.065),(.092,top-.055),(.092,top)]
            b.lathe(f'snooker_leg_{sx}_{i}', (x,y), profile, 'snooker_wood')
            for j,z in enumerate((FLOOR_Z+.044,FLOOR_Z+.398,top-.035)):
                b.lathe(f'snooker_leg_collar_{sx}_{i}_{j}', (x,y),
                        [(.087 if j < 2 else .094,z-.008),(.087 if j < 2 else .094,z+.008)],
                        'snooker_moulding')
            b.lathe(f'snooker_leg_foot_{sx}_{i}', (x,y),
                    [(.066,FLOOR_Z),(.066,FLOOR_Z+.017)],'snooker_iron')
    b.box('snooker_cross_bearer_top', (0, 1.52, SURFACE_Z-.25), (.77,.055,.045), 'snooker_panel')
    b.box('snooker_cross_bearer_baulk', (0, -1.52, SURFACE_Z-.25), (.77,.055,.045), 'snooker_panel')


def add_pockets(b: Builder):
    for index,p in enumerate(POCKETS):
        x,y = p.center
        outward = math.atan2(y/HALF_LENGTH, x/HALF_WIDTH) if p.kind == 'corner' else (0 if x>0 else math.pi)
        sweep = math.radians(116 if p.kind == 'corner' else 106)
        angles = np.linspace(outward-sweep, outward+sweep, 58)
        # Broad rolled leather horseshoe, leaving the playing side open.
        vertices,faces = [],[]
        for theta in angles:
            for phi in np.linspace(0,2*math.pi,12,endpoint=False):
                radius = p.drop_radius+.012+.010*math.cos(phi)
                vertices.append((x+radius*math.cos(theta),y+radius*math.sin(theta),
                                 SURFACE_Z+.029+.010*math.sin(phi)))
        for i in range(len(angles)-1):
            for j in range(12):
                a,c=i*12+j,i*12+(j+1)%12
                faces += [(a,c,c+12),(a,c+12,a+12)]
        b.mesh(f'snooker_pocket_leather_{index}',vertices,faces,'snooker_leather')
        for label,radius,z,thickness,material in (
                ('iron',p.drop_radius+.016,SURFACE_Z-.014,.006,'snooker_iron'),
                ('rim',p.drop_radius-.001,SURFACE_Z-.025,.003,'snooker_net')):
            points=[(x+radius*math.cos(t),y+radius*math.sin(t),z) for t in angles]
            b.lines(f'snooker_pocket_{label}_{index}',list(zip(points,points[1:])),thickness,material)
        stitches=[]
        for theta in angles[::2]:
            r=p.drop_radius+.016
            stitches.append(((x+(r-.002)*math.cos(theta),y+(r-.002)*math.sin(theta),SURFACE_Z+.0384),
                             (x+(r+.002)*math.cos(theta),y+(r+.002)*math.sin(theta),SURFACE_Z+.0384)))
        b.lines(f'snooker_pocket_stitches_{index}',stitches,.00065,'snooker_stitch')
        # Open diamond mesh net, tapering to a sewn leather bottom 29cm below slate.
        rings=[]
        for level in range(11):
            z=SURFACE_Z-.026-level*.026
            r=p.drop_radius*(1-.30*(level/10)**2)
            rings.append([(x+r*math.cos(2*math.pi*(j+.5*(level%2))/20),
                           y+r*math.sin(2*math.pi*(j+.5*(level%2))/20),z) for j in range(20)])
        strands=[]
        for k in range(10):
            for j in range(20):
                strands += [(rings[k][j],rings[k+1][j]),
                            (rings[k][j],rings[k+1][(j+(-1 if k%2==0 else 1))%20])]
        b.lines(f'snooker_pocket_net_{index}',strands,.0012,'snooker_net',sides=5)
        b.lathe(f'snooker_pocket_net_bottom_{index}',(x,y),
                [(p.drop_radius*.72,SURFACE_Z-.30),(p.drop_radius*.72,SURFACE_Z-.286)],'snooker_leather')
        # Two visible pocket-iron mounting screws, outside the ball path.
        for j,theta in enumerate((angles[0]+.10,angles[-1]-.10)):
            b.lathe(f'snooker_pocket_stud_{index}_{j}',
                    (x+(p.drop_radius+.019)*math.cos(theta),y+(p.drop_radius+.019)*math.sin(theta)),
                    [(.003,SURFACE_Z+.037),(.003,SURFACE_Z+.040)],'snooker_brass',count=16)
        E(b.world,'site',name=p.name,pos=numbers((x,y,SURFACE_Z)),size=f'{p.drop_radius:g}',
          group='3',rgba='0 0 0 0')


def add_markings(b: Builder):
    # Flat, non-colliding ink ribbons. A 0.2 mm visual offset separates their
    # depth from the cloth; supersampling keeps the 2 mm width smooth.
    # Shell inertia lets MuJoCo compile these zero-thickness visual meshes.
    half_width, z = .001, SURFACE_Z+.0002
    b.mesh('snooker_baulk',
           [(-HALF_WIDTH, BAULK_Y-half_width, z), (HALF_WIDTH, BAULK_Y-half_width, z),
            (HALF_WIDTH, BAULK_Y+half_width, z), (-HALF_WIDTH, BAULK_Y+half_width, z)],
           [(0, 1, 2), (0, 2, 3)], 'snooker_marking', shell=True)
    # Adjacent quads share their edges, forming one continuous semicircle.
    angles = np.linspace(math.pi, 2*math.pi, 257)
    vertices = [(r*math.cos(t), BAULK_Y+r*math.sin(t), z)
                for t in angles for r in (D_RADIUS-half_width, D_RADIUS+half_width)]
    faces = [(a, a+1, a+3) for a in range(0, len(vertices)-2, 2)]
    faces += [(a, a+3, a+2) for a in range(0, len(vertices)-2, 2)]
    b.mesh('snooker_d', vertices, faces, 'snooker_marking', shell=True)
    for n,p in SPOTS.items():
        E(b.world,'site',name=f'snooker_spot_{NAMES[n]}',type='cylinder',
          pos=numbers((p[0],p[1],SURFACE_Z+.0001)),size='.002 .0001',rgba='.65 .69 .54 1')


def build() -> None:
    root=Element('mujoco',model='snooker_match')
    E(root,'compiler',angle='degree',autolimits='true')
    E(root,'option',timestep=f'{SNOOKER_PHYSICS.timestep:g}',gravity='0 0 -9.81',integrator='Euler',
      solver='Newton',cone='elliptic',iterations='80',tolerance='1e-8')
    visual=E(root,'visual')
    E(visual,'global',offheight='1200',offwidth='1600')
    E(visual,'quality',shadowsize='4096',offsamples='4')
    E(visual,'map',znear='.01',shadowclip='5')
    E(visual,'headlight',ambient='.43 .43 .43',diffuse='.40 .40 .40',specular='.09 .09 .09')
    default=E(root,'default')
    for name,attrs in SNOOKER_PHYSICS.contact_defaults().items():
        contact=E(default,'default',{'class':name})
        extra={'size':f'{BALL_RADIUS:g}','type':'sphere'} if name=='snooker_ball' else {}
        E(contact,'geom',**extra,**attrs)
    asset=E(root,'asset')
    add_materials(asset)
    world=E(root,'worldbody')
    E(world,'geom',name='snooker_floor',type='plane',size='6 6 .01',
      pos=f'0 0 {FLOOR_Z}',rgba='.12 .135 .14 1',contype='0',conaffinity='0')
    # Broad, high overhead lights illuminate both ends of the playing surface.
    for i,(x,y) in enumerate(((-1.4,-1.6),(1.4,-1.6),(-1.4,1.6),(1.4,1.6))):
        E(world,'light',name=f'snooker_light_{i}',pos=f'{x} {y} 5.4',dir=f'{-x*.15} {-y*.10} -1',
          directional='false',diffuse='.62 .62 .60',specular='.14 .14 .13',
          castshadow='true' if i==0 else 'false',attenuation='1 0 0')
    # Side fill reveals the pocket nets, leather and cabinet below the rails.
    for i,(x,y) in enumerate(((3.,-3.4),(-3.,3.4))):
        E(world,'light',name=f'snooker_fill_{i}',pos=f'{x} {y} 2.8',dir=f'{-x} {-y} -1.9',
          directional='false',diffuse='.49 .48 .46',specular='.08 .08 .08',
          castshadow='false',attenuation='1 0 0',cutoff='65',exponent='1')
    # Screen up is world -Y, placing the baulk line and D at the top.
    E(world,'camera',name='snooker_overhead',pos='0 0 5.7',xyaxes='-1 0 0 0 -1 0',fovy='50')
    def camera(name,position,target,fovy):
        forward=np.array(target)-position
        forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1])
        right/=np.linalg.norm(right)
        up=np.cross(right,forward)
        E(world,'camera',name=name,pos=numbers(position),xyaxes=numbers([right,up]),fovy=str(fovy))
    camera('snooker_oblique',(3.,-4.2,3.2),(0.,0.,.50),39)
    camera('snooker_corner_detail',(1.40,-2.40,1.50),(.88,-1.75,.92),42)
    camera('snooker_middle_detail',(1.55,-.36,1.40),(.91,0.,.94),42)

    b=Builder(asset,world)
    add_slate(b)
    add_cushions(b)
    add_cabinet(b)
    add_pockets(b)
    add_markings(b)
    E(root,'include',file='snooker_balls.xml')
    E(root,'include',file='cue_physics.xml')
    indent(root)
    (ROOT/'models/snooker_scene.xml').write_text(tostring(root,encoding='unicode')+'\n')
    balls=Element('mujoco')
    world=E(balls,'worldbody')
    colors={0:'.66 .65 .60 1',16:'.52 .43 .010 1',17:'.014 .18 .045 1',18:'.17 .056 .013 1',
            19:'.013 .065 .35 1',20:'.50 .14 .25 1',21:'.005 .005 .006 1'}
    for n in range(22):
        name='cue_ball' if n==0 else f'object_ball_{n}'
        body=E(world,'body',name=name,pos=f'{3+n*.15} 0 4',gravcomp='1')
        E(body,'freejoint',name=name+'_free')
        E(body,'geom',name=name+'_geom',**{'class':'snooker_ball'},material='snooker_ball_finish',
          rgba=colors.get(n,'.35 .004 .008 1'))
    indent(balls)
    (ROOT/'models/snooker_balls.xml').write_text(tostring(balls,encoding='unicode')+'\n')
    print(f'Wrote snooker_scene.xml: {len(world)} balls, {len(b.world.findall("geom"))} table geoms')


if __name__=='__main__':
    build()
