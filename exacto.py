"""Modelo exacto (PLE) para validar la calidad de las heurísticas en instancias pequeñas.
x[s,d,b,r,k] = 1 si la cirugía s se hace el día d, bloque b, en sala r, con cirujano k."""
import random, sys, time
import numpy as np
from cirugias import Instancia, Solucion, generar, constructivo, busqueda_local, grasp, validar


def exacto(ins, t_lim=120):
    import highspy, numpy as np
    vac = Solucion(ins)
    idx = [(s, *o) for s in ins.S for o in vac.opciones(s)]
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(t_lim))
    n = len(idx)
    for _ in range(n):
        h.addVar(0.0, 1.0)
    h.changeColsCost(n, np.arange(n, dtype=np.int32), -np.ones(n))
    h.changeColsIntegrality(n, np.arange(n, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * n))
    grupos = {}
    for j, (s, d, b, r, k) in enumerate(idx):
        for key in (("s", s), ("r", r, d, b), ("k", k, d, b)):
            grupos.setdefault(key, []).append(j)
    for cols in grupos.values():
        h.addRow(-highspy.kHighsInf, 1.0, len(cols), np.array(cols, dtype=np.int32), np.ones(len(cols)))
    h.run()
    return int(round(-h.getInfo().objective_function_value)), str(h.getModelStatus())


def exacto_lexico(ins, n_fijo=None, t_lim=120):
    """Óptimo lexicográfico: con n_fijo cirugías programadas (por defecto, el máximo),
    minimiza la dispersión  Σ_dep (zonas usadas − 1) sobre los departamentos usados.
    Variables: x[s,d,b,r,k], z[dep,zona] (zona usada por el dep), u[dep] (dep usado).
    min Σ z − Σ u   s.a.  x ≤ z[dep(k), zona(r)],  u[dep] ≤ Σ_zona z[dep,zona]."""
    import highspy
    if n_fijo is None:
        n_fijo, _ = exacto(ins, t_lim)
    vac = Solucion(ins)
    idx = [(s, *o) for s in ins.S for o in vac.opciones(s)]
    dep = lambda k: ins.dept_of.get(k, "_sin_dep")
    deps = sorted({dep(k) for k in ins.K}); zs = sorted(set(ins.zone_of.values()))
    nx = len(idx)
    zi = {(d, z): nx + j for j, (d, z) in enumerate((d, z) for d in deps for z in zs)}
    ui = {d: nx + len(zi) + j for j, d in enumerate(deps)}
    n = nx + len(zi) + len(ui)
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(t_lim))
    for _ in range(n):
        h.addVar(0.0, 1.0)
    cols = np.arange(n, dtype=np.int32)
    h.changeColsIntegrality(n, cols, np.array([highspy.HighsVarType.kInteger] * n))
    c = np.zeros(n); c[list(zi.values())] = 1.0; c[list(ui.values())] = -1.0
    h.changeColsCost(n, cols, c)
    inf = highspy.kHighsInf
    g = {}
    for j, (s, d, b, r, k) in enumerate(idx):
        for key in (("s", s), ("r", r, d, b), ("k", k, d, b)):
            g.setdefault(key, []).append(j)
        h.addRow(-inf, 0.0, 2, np.array([j, zi[(dep(k), ins.zone_of[r])]], dtype=np.int32),
                 np.array([1.0, -1.0]))
    for v in g.values():
        h.addRow(-inf, 1.0, len(v), np.array(v, dtype=np.int32), np.ones(len(v)))
    for d in deps:
        cz = [zi[(d, z)] for z in zs]
        h.addRow(-inf, 0.0, len(cz) + 1, np.array([ui[d]] + cz, dtype=np.int32),
                 np.array([1.0] + [-1.0] * len(cz)))
    h.addRow(n_fijo, n_fijo, nx, np.arange(nx, dtype=np.int32), np.ones(nx))
    h.run()
    return n_fijo, int(round(h.getInfo().objective_function_value)), str(h.getModelStatus())


def constructivo_aleatorio(ins, rng):
    """Constructivo ingenuo (orden aleatorio, primera opción libre): punto de partida débil
    para medir cuánto mejora la búsqueda local por sí sola."""
    sol = Solucion(ins)
    orden = ins.S[:]; rng.shuffle(orden)
    for s in orden:
        op = sol.opciones(s)
        if op:
            sol.place(s, rng.choice(op))
    sol.log.clear()
    return sol


class A: grasp, alpha, seed = 20, 0.3, 1

if __name__ == "__main__":
    print(f"{'inst':6}{'|S|':>4}{'Óptimo':>8} | {'Aleat':>6}{'Aleat+LS':>9} | {'Voraz':>6}{'Voraz+LS':>9}{'GRASP':>7}")
    tam = [(30, 6, 10), (40, 5, 10), (60, 6, 12), (80, 6, 14), (100, 8, 16)]
    for i, (ns, nr, nk) in enumerate(tam):
        ins = Instancia(generar(ns, nr, nk, seed=i))
        opt, st = exacto(ins)
        rng = random.Random(7)
        al = constructivo_aleatorio(ins, rng); n_al = al.n_prog()
        als = busqueda_local(al)
        v = constructivo(ins); n_v = v.n_prog()
        vl = busqueda_local(v)
        g = grasp(ins, 20, 0.3, 1)
        for s in (als, vl, g):
            assert not validar(s)
        print(f"sint{i:<2}{ns:>4}{opt:>8} | {n_al:>6}{als.n_prog():>9} | {n_v:>6}{vl.n_prog():>9}{g.n_prog():>7}   ({st})")