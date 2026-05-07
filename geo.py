import argparse
import math
import numpy as np


# Reference mesh size parameters at Re=1000 and Re=3900,
# used for power-law interpolation across the parameter space.
_REF_LOW = {
    "Re":      1000,
    "l_theta": np.pi / 50,
    "l_t":     0.051,
    "l_wb":    0.287,
    "l_ff":    0.9,
}

_REF_HIGH = {
    "Re":      3900,
    "l_theta": np.pi / 116,
    "l_t":     0.022,
    "l_wb":    0.22,
    "l_ff":    0.868,
}


# ---------------------------------------------------------------------------
# Mesh-size utilities
# ---------------------------------------------------------------------------

def get_rotated_superellipse_extremes(a, b, n, theta, resolution=1000):
    """Return the (leftmost, rightmost, topmost, bottommost) points of a
    rotated superellipse with semi-axes a, b, exponent n, and rotation angle theta."""
    max_x = min_x =  0.0
    max_y = min_y =  0.0
    rightmost = leftmost = topmost = bottommost = [0.0, 0.0]

    for i in range(resolution):
        t  = 2 * math.pi * i / resolution
        cx = math.cos(t)
        sy = math.sin(t)

        x = a * (abs(cx)) ** (2 / n) * (1 if cx >= 0 else -1)
        y = b * (abs(sy)) ** (2 / n) * (1 if sy >= 0 else -1)

        x_rot = x * math.cos(theta) - y * math.sin(theta)
        y_rot = x * math.sin(theta) + y * math.cos(theta)

        if i == 0 or x_rot > max_x:
            max_x     = x_rot
            rightmost = [x_rot, y_rot]
        if i == 0 or x_rot < min_x:
            min_x    = x_rot
            leftmost = [x_rot, y_rot]
        if i == 0 or y_rot > max_y:
            max_y   = y_rot
            topmost = [x_rot, y_rot]
        if i == 0 or y_rot < min_y:
            min_y      = y_rot
            bottommost = [x_rot, y_rot]

    return leftmost, rightmost, topmost, bottommost


def power_law_interpolate(x1, y1, x2, y2, x):
    """Interpolate y at x using a power law fitted through (x1,y1) and (x2,y2)."""
    b = np.log(y2 / y1) / np.log(x2 / x1)
    a = y1 / (x1 ** b)
    return a * (x ** b)


def gen_cell_sizes(Re):
    """Return (l_theta, l_t, l_wb, l_ff) interpolated for the given Reynolds number."""
    keys = ["l_theta", "l_t", "l_wb", "l_ff"]
    return tuple(
        power_law_interpolate(_REF_LOW["Re"], _REF_LOW[k], _REF_HIGH["Re"], _REF_HIGH[k], Re)
        for k in keys
    )


def calculate_bl_cells(delta, h1, r):
    """
    Return the number of boundary-layer cells required to span thickness delta,
    given a first-cell height h1 and geometric growth rate r > 1.
    """
    numerator   = math.log(1 + (delta * (r - 1)) / h1)
    denominator = math.log(r)
    return math.ceil(numerator / denominator)


# ---------------------------------------------------------------------------
# Geo-file generation helpers
# ---------------------------------------------------------------------------

def generate_parameters(l_theta, l_t, l_wb, l_ff, GR_BL, t_BL, a, b, n, inflow_angle):
    if n == 1:
        L_longside = L_shortside = np.sqrt(a**2 + b**2)
    elif n == 2:
        # Ramanujan's approximation for the perimeter of an ellipse (quarter arc)
        L_longside = L_shortside = np.pi / 4 * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    elif n >= 5:
        L_longside  = 2 * a
        L_shortside = 2 * b

    lines = [
        "// Gmsh project created by geo.py",
        'SetFactory("OpenCASCADE");',
        "",
        "// =======================",
        "// Parameters",
        "// =======================",
        "",
        "// Extrusion",
        "z_len = 2*Pi;",
        "n_z   = 30;",
        "",
        "// Near-object meshing parameters",
        f"lc_wall = {l_theta};",
        f"t_first = {l_t};",
        f"lc_wake = {l_wb};",
        f"lc_ff   = {l_ff};",
        f"GR_BL   = {GR_BL};",
        f"t_BL    = {t_BL};",
        f"n_BL    = {calculate_bl_cells(t_BL, l_t, GR_BL)};",
        "",
        f"n_longside  = {L_longside / l_theta};",
        f"n_shortside = {L_shortside / l_theta};",
        "",
        f"inflow_angle = {inflow_angle};",
        "",
        "// =======================",
        "// Define 2D geometry",
        "// =======================",
        "",
    ]
    return "\n".join(lines)


def generate_lame_quadrant(a, b, n, N, point_start_id, point_end_id, spline_id,
                           quadrant_nr, hoek_start, hoek_length, inflow_angle):
    loop_lines = [
        f"For i In {{0 : {N}}}",
        f"For i In {{1 : {N}}}",
        f"For i In {{1 : {N}}}",
        f"For i In {{1 : {N - 1}}}",
    ]

    if n == np.inf:
        corners_start = [[a, b], [-a, b], [-a, -b], [a, -b]]
        corners_end   = [[-a, b], [-a, -b], [a, -b], [a, b]]
        cos_a, sin_a  = math.cos(inflow_angle), math.sin(inflow_angle)

        def rotate(pts):
            return [[p[0] * cos_a - p[1] * sin_a, p[0] * sin_a + p[1] * cos_a] for p in pts]

        rot_start = rotate(corners_start)
        rot_end   = rotate(corners_end)
        q         = quadrant_nr - 1

        start_decl = [
            f"Point({point_start_id}) = {{{rot_start[q][0]}, {rot_start[q][1]}, 0}};",
            "", "", "",
        ]
        end_decl = [
            f"Point({point_end_id}) = {{{rot_end[q][0]}, {rot_end[q][1]}, 0}};",
            f"Point({point_end_id}) = {{{rot_end[q][0]}, {rot_end[q][1]}, 0}};",
            f"Point({point_end_id}) = {{{rot_end[q][0]}, {rot_end[q][1]}, 0}};",
            "",
        ]
        lame_lines = [
            "",
            "// Create a quadrant of the superellipse",
            start_decl[q],
            end_decl[q],
            f"Line({spline_id}) = {{{point_start_id}, {point_end_id}}};",
        ]
    else:
        lame_lines = [
            "",
            "// Create a quadrant of the superellipse",
            loop_lines[quadrant_nr - 1],
            f"    t = {hoek_start} + {hoek_length} * i / {N};",
            "    cx = Cos(t);",
            "    sy = Sin(t);",
            f"    x = {a} * (Fabs(cx))^(2/{n}) * (cx < 0 ? -1 : 1);",
            f"    y = {b} * (Fabs(sy))^(2/{n}) * (sy < 0 ? -1 : 1);",
            f"    x_rot = x * Cos(inflow_angle) - y * Sin(inflow_angle);",
            f"    y_rot = x * Sin(inflow_angle) + y * Cos(inflow_angle);",
            f"    Point(i + {point_start_id}) = {{x_rot, y_rot, 0}};",
            "EndFor",
            f"Spline({spline_id}) = {{{point_start_id} : {point_start_id + N - 1}, {point_end_id}}};",
        ]

    return "\n".join(lame_lines)


def generate_loop(loop_id, curves):
    lines = [
        "",
        "// Combine separate curves into a loop",
        f"Curve Loop({loop_id}) = {{{', '.join(map(str, curves))}}};",
        "",
    ]
    return "\n".join(lines)


def get_hoeken(a, b, n, quadrant):
    Os = [b, b, -b, -b]
    As = [a, -a, -a, a]
    if n >= 5:
        hoek1 = np.arctan2(Os[quadrant - 1], As[quadrant - 1]) % (2 * np.pi)
        hoek2 = np.arctan2(Os[quadrant % 4], As[quadrant % 4]) % (2 * np.pi)
        if hoek2 < hoek1:
            hoek2 += 2 * np.pi
        hoek_length = hoek2 - hoek1
    else:
        hoek1       = (quadrant - 1) * np.pi / 2
        hoek_length = np.pi / 2
    return hoek1, hoek_length


def generate_lame(a, b, n, N, pstart, pends, spline_ids, loop_id, inflow_angle):
    templines = []
    for j in range(4):
        hoek1, hoek_length = get_hoeken(a, b, n, j + 1)
        templines += generate_lame_quadrant(
            a=a, b=b, n=n, N=N,
            point_start_id=pstart + j * N,
            point_end_id=pends[j],
            spline_id=spline_ids[j],
            quadrant_nr=j + 1,
            hoek_start=hoek1,
            hoek_length=hoek_length,
            inflow_angle=inflow_angle,
        ).splitlines()
    templines += generate_loop(loop_id, spline_ids).splitlines()
    return "\n".join(templines)


def connect_object_BL(object_point_ids, bl_point_ids,
                      object_spline_ids, bl_spline_ids,
                      line_ids, block_ids):
    lines = ["", "// Connect object to boundary-layer curves"]
    for i in range(len(object_point_ids)):
        lines.append(f"Line({line_ids[i]}) = {{{object_point_ids[i]}, {bl_point_ids[i]}}};")
    lines.append("")
    for i in range(len(object_spline_ids)):
        j = i + 1 if i < len(object_spline_ids) - 1 else 0
        lines += [
            f"Curve Loop({block_ids[i]}) = {{{object_spline_ids[i]}, {line_ids[j]}, -{bl_spline_ids[i]}, -{line_ids[i]}}};",
            f"Plane Surface({block_ids[i]}) = {{{block_ids[i]}}};",
        ]
    lines += [
        "",
        f"Transfinite Curve {{{line_ids[0]}, {line_ids[1]}, {line_ids[2]}, {line_ids[3]}}} = n_BL Using Progression GR_BL;",
        f"Transfinite Curve {{{object_spline_ids[0]}, {object_spline_ids[2]}, {bl_spline_ids[0]}, {bl_spline_ids[2]}}} = n_longside;",
        f"Transfinite Curve {{{object_spline_ids[1]}, {object_spline_ids[3]}, {bl_spline_ids[1]}, {bl_spline_ids[3]}}} = n_shortside;",
        f"Transfinite Surface {{{block_ids[0]}, {block_ids[1]}, {block_ids[2]}, {block_ids[3]}}};",
        f"Recombine Surface {{{block_ids[0]}, {block_ids[1]}, {block_ids[2]}, {block_ids[3]}}};",
    ]
    return "\n".join(lines)


def generate_wakebox(l_wb, r_wb, t_wb, b_wb, wb_ids, loop_id):
    lines = [
        "",
        "// Define wake box for refinement",
        f"Point({wb_ids[0]}) = {{{r_wb}, {t_wb}, 0}};",
        f"Point({wb_ids[1]}) = {{{l_wb}, {t_wb}, 0}};",
        f"Point({wb_ids[2]}) = {{{l_wb}, {b_wb}, 0}};",
        f"Point({wb_ids[3]}) = {{{r_wb}, {b_wb}, 0}};",
        f"Line({wb_ids[0]}) = {{{wb_ids[0]}, {wb_ids[1]}}};",
        f"Line({wb_ids[1]}) = {{{wb_ids[1]}, {wb_ids[2]}}};",
        f"Line({wb_ids[2]}) = {{{wb_ids[2]}, {wb_ids[3]}}};",
        f"Line({wb_ids[3]}) = {{{wb_ids[3]}, {wb_ids[0]}}};",
        f"Curve Loop({loop_id}) = {{{wb_ids[0]}, {wb_ids[1]}, {wb_ids[2]}, {wb_ids[3]}}};",
        "",
    ]
    return "\n".join(lines)


def generate_domain(l_wb, r_ff, xmax, domain_ids, loop_id=70):
    shift = l_wb + 2
    lines = [
        "",
        "// Define far-field domain",
        f"Point({domain_ids[0]}) = {{{shift}, {r_ff}, 0}};",
        f"Point({domain_ids[1]}) = {{{-r_ff + shift}, 0, 0}};",
        f"Point({domain_ids[2]}) = {{{shift}, {-r_ff}, 0}};",
        f"Point({domain_ids[3]}) = {{{xmax}, {-r_ff}, 0}};",
        f"Point({domain_ids[4]}) = {{{xmax}, {r_ff}, 0}};",
        f"Point({domain_ids[5]}) = {{{shift}, 0, 0}};",
        f"Circle({domain_ids[0]}) = {{{domain_ids[0]}, {domain_ids[5]}, {domain_ids[1]}}};",
        f"Circle({domain_ids[1]}) = {{{domain_ids[1]}, {domain_ids[5]}, {domain_ids[2]}}};",
        f"Line({domain_ids[2]}) = {{{domain_ids[2]}, {domain_ids[3]}}};",
        f"Line({domain_ids[3]}) = {{{domain_ids[3]}, {domain_ids[4]}}};",
        f"Line({domain_ids[4]}) = {{{domain_ids[4]}, {domain_ids[0]}}};",
        "",
        f"Curve Loop({loop_id}) = {{{domain_ids[0]}, {domain_ids[1]}, {domain_ids[2]}, {domain_ids[3]}, {domain_ids[4]}}};",
    ]
    return "\n".join(lines)


def create_remaining_surfaces(surf_ids, bl_loop_id, wb_loop_id, domain_loop_id):
    lines = [
        "",
        "// Create remaining surfaces for meshing",
        f"Plane Surface({surf_ids[0]}) = {{{wb_loop_id}, {bl_loop_id}}};",
        f"Plane Surface({surf_ids[1]}) = {{{domain_loop_id}, {wb_loop_id}}};",
    ]
    return "\n".join(lines)


def extrude(surf_ids, block_ids):
    lines = [
        "",
        "// =================================================",
        "// Create 3D domain",
        "// =================================================",
        f"out[] = Extrude {{0, 0, z_len}} {{",
        f"  Surface{{{surf_ids[0]}, {surf_ids[1]}, {block_ids[0]}, {block_ids[1]}, {block_ids[2]}, {block_ids[3]}}};",
        f"  Layers{{n_z}};",
        f"  Recombine;",
        "};",
        "",
        "// Periodic boundary links",
        f"Periodic Surface {{out[0]}}  = {{{surf_ids[0]}}}  Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[10]}} = {{{surf_ids[1]}}}  Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[21]}} = {{{block_ids[0]}}} Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[27]}} = {{{block_ids[1]}}} Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[33]}} = {{{block_ids[2]}}} Translate {{0,0,z_len}};",
        f"Periodic Surface {{out[39]}} = {{{block_ids[3]}}} Translate {{0,0,z_len}};",
        "",
        "// Physical groups",
        f'Physical Volume("Fluid") = {{out[1], out[11], out[22], out[28], out[34], out[40]}};',
        f'Physical Surface("inlet")  = {{out[12], out[13]}};',
        f'Physical Surface("outlet") = {{out[14], out[15], out[16]}};',
        f'Physical Surface("wall")   = {{out[23], out[29], out[35], out[41]}};',
        f'Physical Surface("periodic-0-l") = {{{surf_ids[0]}, {surf_ids[1]}, {block_ids[0]}, {block_ids[1]}, {block_ids[2]}, {block_ids[3]}}};',
        f'Physical Surface("periodic-0-r") = {{out[0], out[10], out[21], out[27], out[33], out[39]}};',
    ]
    return "\n".join(lines)


def generate_meshing(wb_ids, surf_ids, object_spline_ids, t_wb):
    wbstring              = ", ".join(str(i) for i in wb_ids)
    object_spline_str     = ", ".join(str(i) for i in object_spline_ids)
    lines = [
        "",
        "// =================================================",
        "// Meshing parameters",
        "// =================================================",
        "",
        "// Wake refinement",
        "Field[1] = Distance;",
        f"Field[1].CurvesList = {{{wbstring}}};",
        "Field[1].Sampling = 100;",
        "",
        "Field[2] = Threshold;",
        "Field[2].InField = 1;",
        "Field[2].SizeMin = lc_wake;",
        "Field[2].SizeMax = lc_ff;",
        f"Field[2].DistMin = {t_wb} + 0.01;",
        f"Field[2].DistMax = {t_wb} + 0.02;",
        "",
        "Field[3] = Restrict;",
        "Field[3].InField = 2;",
        f"Field[3].SurfacesList = {{{surf_ids[0]}}};",
        "Field[3].VolumesList = {out[1]};",
        "",
        "// Cylinder-wall refinement",
        "Field[4] = Distance;",
        f"Field[4].CurvesList = {{{object_spline_str}}};",
        "Field[4].Sampling = 100;",
        "",
        "Field[5] = Threshold;",
        "Field[5].InField = 4;",
        "Field[5].SizeMin = lc_wall;",
        "Field[5].SizeMax = lc_ff;",
        "Field[5].DistMin = t_BL;",
        "Field[5].DistMax = t_BL + 0.5;",
        "",
        "// Combine fields",
        "Field[7] = Min;",
        "Field[7].FieldsList = {3, 5};",
        "Background Field = 7;",
        "",
        "// Algorithm settings",
        "Mesh.Algorithm = 6;",
        "Mesh.ElementOrder = 4;",
        "Mesh.HighOrderOptimize = 0;",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def generate_geo(l_theta, l_t, l_wb, l_ff, a, b, n, GR_BL, t_BL, inflow_angle):
    lines = generate_parameters(
        l_theta=l_theta, l_t=l_t, l_wb=l_wb, l_ff=l_ff,
        GR_BL=GR_BL, t_BL=t_BL, a=a, b=b, n=n, inflow_angle=inflow_angle,
    ).splitlines()

    object_spline_ids = [1, 2, 3, 4]
    pstart = 1000
    N      = 100
    pends_object = [pstart + N, pstart + 2 * N, pstart + 3 * N, pstart]

    lines += generate_lame(
        a=a, b=b, n=n, N=N, pstart=pstart, pends=pends_object,
        spline_ids=object_spline_ids, loop_id=10, inflow_angle=inflow_angle,
    ).splitlines()

    # Boundary-layer shape (slightly expanded superellipse)
    a_BL = a + t_BL
    b_BL = b + t_BL
    n_BL_map = {1: 1.5, 2: 2, 5: 5}
    n_BL = n_BL_map.get(n, 10 if n == np.inf else n)

    bl_spline_ids = [11, 12, 13, 14]
    pstart_BL     = 4000
    bl_loop_id    = 20
    pends_BL      = [pstart_BL + N, pstart_BL + 2 * N, pstart_BL + 3 * N, pstart_BL]

    lines += generate_lame(
        a=a_BL, b=b_BL, n=n_BL, N=N, pstart=pstart_BL, pends=pends_BL,
        spline_ids=bl_spline_ids, loop_id=bl_loop_id, inflow_angle=inflow_angle,
    ).splitlines()

    object_point_ids   = [pends_object[-1]] + pends_object[:-1]
    bl_point_ids       = [pends_BL[-1]] + pends_BL[:-1]
    connection_line_ids = [31, 32, 33, 34]
    block_ids           = [41, 42, 43, 44]

    lines += connect_object_BL(
        object_point_ids, bl_point_ids,
        object_spline_ids, bl_spline_ids,
        connection_line_ids, block_ids,
    ).splitlines()

    leftmost, rightmost, topmost, bottommost = get_rotated_superellipse_extremes(
        a_BL, b_BL, n_BL, inflow_angle
    )
    r_wb = rightmost[0] + 7
    l_wb = leftmost[0]  - 1
    t_wb = topmost[1]   + 1
    b_wb = bottommost[1] - 1

    wb_ids     = [51, 52, 53, 54]
    wb_loop_id = 60
    lines += generate_wakebox(l_wb, r_wb, t_wb, b_wb, wb_ids, loop_id=wb_loop_id).splitlines()

    r_ff        = 9
    xmax        = 25
    domain_ids  = [61, 62, 63, 64, 65, 66]
    domain_loop_id = 70
    lines += generate_domain(l_wb, r_ff, xmax, domain_ids, loop_id=domain_loop_id).splitlines()

    surf_ids = [101, 201]
    lines += create_remaining_surfaces(surf_ids, bl_loop_id, wb_loop_id, domain_loop_id).splitlines()
    lines += extrude(surf_ids, block_ids).splitlines()
    lines += generate_meshing(wb_ids, surf_ids, object_spline_ids, t_wb).splitlines()

    return "\n".join(lines)


def main(Re, inflow_angle, a, b, n, GR_BL, t_BL, geo_output=None):
    l_theta, l_t, l_wb, l_ff = gen_cell_sizes(Re)
    content = generate_geo(
        l_theta=l_theta, l_t=l_t, l_wb=l_wb, l_ff=l_ff,
        a=a, b=b, n=n, GR_BL=GR_BL, t_BL=t_BL,
        inflow_angle=inflow_angle,
    )
    if geo_output:
        with open(geo_output, "w") as f:
            f.write(content)
    else:
        print(content)


if __name__ == "__main__":
    main(
        Re=3500,
        inflow_angle=40 * np.pi / 180,
        a=1.5,
        b=0.5,
        n=np.inf,
        GR_BL=1.021,
        t_BL=0.5,
        geo_output="cylinder.geo",
    )
