"""
Taller 1 - Metaheurística (UdeA) - Programación semanal de cirugías
Punto 1: Constructivo voraz/GRASP con orden dinámico "más restringida primero"
Punto 3: Búsqueda local VND (inserción, cadenas de expulsión, reubicación, intercambio de salas)

Uso:
    python cirugias.py instancia.json [--grasp N] [--alpha 0.3] [--seed 1]
    python cirugias.py --demo            # instancias sintéticas con la misma estructura
"""
import json, random, sys, time, argparse
from collections import defaultdict


# ----------------------------------------------------------------------------- datos
def cargar_instancia(path):
    """Acepta JSON con las claves del enunciado, o un .py/.txt con asignaciones
    del tipo  Surgeries = [...]  Type_Surgery = {...}"""
    txt = open(path, encoding="utf-8").read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        ns = {}
        exec(txt, {}, ns)
        return ns


class Instancia:
    def __init__(self, D):
        self.S = list(D["Surgeries"]); self.R = list(D["Rooms"])
        self.Days = list(D["Days"]); self.Blocks = list(D["Blocks"])
        self.K = list(D["Surgeons"])
        self.type_of = D["Type_Surgery"]
        self.zone_of = D["Rooms_Z"]
        self.days_s = {s: set(D["Days_Surgery"][s]) for s in self.S}
        self.days_k = {k: set(D["Days_Surgeon"].get(k, [])) for k in self.K}
        rooms_type = D["Rooms_Type"]; type_k = D["Type_Surgeon"]
        # compatibilidades precalculadas por cirugía
        self.rooms_for = {s: list(rooms_type.get(self.type_of[s], [])) for s in self.S}
        self.surg_for = {s: [k for k in self.K if self.type_of[s] in type_k.get(k, [])]
                         for s in self.S}
        # departamento de cada cirujano (si aparece en varios, se toma el primero)
        self.dept_of = {}
        for dep, ks in D["Dept_Surgeon"].items():
            for k in ks:
                self.dept_of.setdefault(k, dep)
        self.Dept_Surgeon = D["Dept_Surgeon"]


# ----------------------------------------------------------------------------- solución
class Solucion:
    """x[s] = (d, b, r, k) o None.  Mantiene mapas de ocupación para chequeos O(1)."""
    def __init__(self, ins):
        self.ins = ins
        self.x = {s: None for s in ins.S}
        self.room = {}          # (r, d, b) -> s
        self.surg = {}          # (k, d, b) -> s
        self.log = []           # bitácora para deshacer (cadenas de expulsión)

    # --- primitivas con bitácora
    def place(self, s, o):
        d, b, r, k = o
        self.x[s] = o; self.room[(r, d, b)] = s; self.surg[(k, d, b)] = s
        self.log.append(("p", s, o))

    def remove(self, s):
        d, b, r, k = o = self.x[s]
        self.x[s] = None; del self.room[(r, d, b)]; del self.surg[(k, d, b)]
        self.log.append(("r", s, o))

    def rollback(self, mark):
        while len(self.log) > mark:
            op, s, o = self.log.pop()
            d, b, r, k = o
            if op == "p":
                self.x[s] = None; del self.room[(r, d, b)]; del self.surg[(k, d, b)]
            else:
                self.x[s] = o; self.room[(r, d, b)] = s; self.surg[(k, d, b)] = s

    # --- opciones
    def opciones(self, s, ignorar_ocupacion=False):
        """Todas las (d,b,r,k) factibles para s (respetando ocupación salvo que se ignore)."""
        ins = self.ins; out = []
        for d in ins.days_s[s]:
            ks = [k for k in ins.surg_for[s] if d in ins.days_k[k]]
            if not ks:
                continue
            for b in ins.Blocks:
                for r in ins.rooms_for[s]:
                    if not ignorar_ocupacion and (r, d, b) in self.room:
                        continue
                    for k in ks:
                        if not ignorar_ocupacion and (k, d, b) in self.surg:
                            continue
                        out.append((d, b, r, k))
        return out

    # --- objetivo
    def n_prog(self):
        return sum(1 for o in self.x.values() if o is not None)

    def dispersion(self):
        """Σ_dep (nº de zonas usadas por el departamento − 1). 0 = cada dep en una sola zona."""
        z = defaultdict(set)
        for s, o in self.x.items():
            if o is not None:
                z[self.ins.dept_of.get(o[3])].add(self.ins.zone_of[o[2]])
        return sum(len(v) - 1 for v in z.values())

    def minoritarias(self):
        """Nº de cirugías fuera de la zona mayoritaria de su departamento. A diferencia de
        dispersion(), cambia con cada movimiento individual: da gradiente a la búsqueda local."""
        return sum(sum(c.values()) - max(c.values()) for c in self.zonas_dep().values())

    def objetivo(self):
        # lexicográfico: nº de cirugías, luego menor dispersión, luego menos cirugías minoritarias
        return (self.n_prog(), -self.dispersion(), -self.minoritarias())

    def copia(self):
        c = Solucion(self.ins)
        c.x = dict(self.x); c.room = dict(self.room); c.surg = dict(self.surg)
        return c

    def zonas_dep(self):
        z = defaultdict(lambda: defaultdict(int))
        for s, o in self.x.items():
            if o is not None:
                z[self.ins.dept_of.get(o[3])][self.ins.zone_of[o[2]]] += 1
        return z


# ----------------------------------------------------------------------------- validación
def validar(sol):
    ins = sol.ins; err = []; vr = set(); vk = set()
    for s, o in sol.x.items():
        if o is None:
            continue
        d, b, r, k = o
        if d not in ins.days_s[s]: err.append(f"{s}: día {d} no permitido")
        if b not in ins.Blocks: err.append(f"{s}: bloque {b} inválido")
        if r not in ins.rooms_for[s]: err.append(f"{s}: sala {r} no equipada")
        if k not in ins.surg_for[s]: err.append(f"{s}: cirujano {k} no certificado")
        if d not in ins.days_k[k]: err.append(f"{s}: cirujano {k} no disponible {d}")
        if (r, d, b) in vr: err.append(f"sala {r} doble en {d}-{b}")
        if (k, d, b) in vk: err.append(f"cirujano {k} doble en {d}-{b}")
        vr.add((r, d, b)); vk.add((k, d, b))
    return err


# ----------------------------------------------------------------------------- Punto 1
def constructivo(ins, alpha=0.0, rng=None, lam=0.5, sol=None):
    """Voraz (alpha=0) o GRASP (alpha>0).
    1) Selección dinámica de la cirugía con MENOS opciones factibles (MRV / tipo DSatur).
    2) Para ella, se elige la opción de menor costo:
         presión(r,d,b) + presión(k,d,b)  +  lam * penalización_de_zona
       presión = nº de cirugías pendientes que aún podrían usar ese recurso en ese bloque
       (se reserva lo escaso para quien más lo necesita).
       penalización de zona = 1 si la sala está en una zona que el departamento del cirujano
       aún no usa (y ya usa alguna otra).
    Si se pasa `sol` (parcial), completa solo las cirugías sin programar: así sirve de
    operador de reparación en LNS.
    """
    rng = rng or random.Random(0)
    sol = sol if sol is not None else Solucion(ins)
    pendientes = {s for s in ins.S if sol.x[s] is None}
    while pendientes:
        opts = {s: sol.opciones(s) for s in pendientes}
        for s in [s for s in pendientes if not opts[s]]:
            pendientes.discard(s)            # imposible de programar: queda sin asignar
        if not pendientes:
            break
        # demanda potencial sobre cada recurso-bloque
        dem = defaultdict(int)
        for s in pendientes:
            for d, b, r, k in opts[s]:
                dem[("R", r, d, b)] += 1; dem[("K", k, d, b)] += 1
        # cirugía más restringida (desempate aleatorio en GRASP)
        s = min(pendientes, key=lambda t: (len(opts[t]), rng.random() if alpha > 0 else 0, t))
        zd = sol.zonas_dep()

        def costo(o):
            d, b, r, k = o
            dep = ins.dept_of.get(k); zonas = zd.get(dep, {})
            pen = 1 if zonas and ins.zone_of[r] not in zonas else 0
            return dem[("R", r, d, b)] + dem[("K", k, d, b)] + lam * pen

        cs = [(costo(o), o) for o in opts[s]]
        cmin = min(c for c, _ in cs); cmax = max(c for c, _ in cs)
        umbral = cmin + alpha * (cmax - cmin)
        rcl = [o for c, o in cs if c <= umbral + 1e-9]
        sol.place(s, rng.choice(rcl) if alpha > 0 else min(rcl))
        pendientes.discard(s)
    sol.log.clear()
    return sol


# ----------------------------------------------------------------------------- Punto 3
def insertar(sol, s, prof, prohibidas):
    """Intenta programar s. Si no hay hueco libre, usa una cadena de expulsión:
    ocupa un hueco 'ocupado', expulsa a quien(es) estorban (máx. 2: la de la sala y la del
    cirujano) y las reinserta recursivamente con profundidad prof-1."""
    libres = sol.opciones(s)
    if libres:
        sol.place(s, min(libres)); return True
    if prof == 0:
        return False
    for o in sol.opciones(s, ignorar_ocupacion=True):
        d, b, r, k = o
        conf = {sol.room.get((r, d, b)), sol.surg.get((k, d, b))} - {None}
        if conf & prohibidas:
            continue
        mark = len(sol.log)
        for c in conf:
            sol.remove(c)
        sol.place(s, o)
        if all(insertar(sol, c, prof - 1, prohibidas | {s}) for c in conf):
            return True
        sol.rollback(mark)
    return False


def N_insercion(sol, prof):
    """Vecindario de inserción / cadenas de expulsión: busca +1 cirugía programada."""
    for s in [s for s, o in sol.x.items() if o is None]:
        mark = len(sol.log)
        if insertar(sol, s, prof, frozenset()):
            return True
        sol.rollback(mark)
    return False


def N_reubicar(sol):
    """Mueve una cirugía a otra opción libre (día/bloque/sala/cirujano, incluso a un cirujano de
    otro departamento) si mejora (dispersión, minoritarias)."""
    base = (sol.dispersion(), sol.minoritarias())
    for s, o in list(sol.x.items()):
        if o is None:
            continue
        mark = len(sol.log); sol.remove(s)
        for o2 in sol.opciones(s):
            sol.place(s, o2)
            if (sol.dispersion(), sol.minoritarias()) < base:
                return True
            sol.rollback(len(sol.log) - 1)
        sol.rollback(mark)
    return False


def N_swap_salas(sol):
    """Intercambia las salas de dos cirugías del mismo día-bloque (si ambas son compatibles)."""
    base = (sol.dispersion(), sol.minoritarias()); ins = sol.ins
    prog = [s for s, o in sol.x.items() if o is not None]
    for i, a in enumerate(prog):
        da, ba, ra, ka = sol.x[a]
        for c in prog[i + 1:]:
            dc, bc, rc, kc = sol.x[c]
            if (da, ba) != (dc, bc) or ins.zone_of[ra] == ins.zone_of[rc]:
                continue
            if rc in ins.rooms_for[a] and ra in ins.rooms_for[c]:
                mark = len(sol.log)
                sol.remove(a); sol.remove(c)
                sol.place(a, (da, ba, rc, ka)); sol.place(c, (dc, bc, ra, kc))
                if (sol.dispersion(), sol.minoritarias()) < base:
                    return True
                sol.rollback(mark)
    return False


def N_sacar1_meter2(sol, prof=1):
    """Saca una cirugía programada c y trata de programar DOS de las no programadas
    (con cadenas de expulsión de profundidad `prof`). Ganancia neta +1.
    Cubre el caso que las cadenas no ven: el óptimo deja por fuera a una cirugía distinta."""
    sin = [s for s, o in sol.x.items() if o is None]
    if len(sin) < 1:
        return False
    for c in [s for s, o in sol.x.items() if o is not None]:
        mark = len(sol.log); sol.remove(c)
        metidas = 0
        for u in sin:
            m2 = len(sol.log)
            if insertar(sol, u, prof, frozenset({c})):
                metidas += 1
                if metidas == 2:
                    return True
            else:
                sol.rollback(m2)
        sol.rollback(mark)
    return False


def lns(sol, iters=200, k=4, alpha=0.2, seed=0, t_lim=20.0):
    """Large Neighborhood Search (destruir y reparar) sobre una solución ya pulida por VND.
    Destruir: se escoge una cirugía no programada u (o una al azar si todas están programadas)
    y se retiran hasta k cirugías que ocupan salas o cirujanos que u podría usar.
    Reparar: el mismo constructivo (MRV + RCL con alpha) completa la solución, y luego VND.
    Aceptación: si no empeora (permite moverse por mesetas); se guarda la mejor."""
    rng = random.Random(seed); ins = sol.ins
    cur = sol.copia(); best = cur.copia(); t0 = time.time()
    for _ in range(iters):
        if time.time() - t0 > t_lim:
            break
        s = cur.copia()
        sin = [x for x, o in s.x.items() if o is None]
        foco = rng.choice(sin) if sin else rng.choice(ins.S)
        rel = set()
        for d, b, r, kk in s.opciones(foco, ignorar_ocupacion=True):
            for occ in (s.room.get((r, d, b)), s.surg.get((kk, d, b))):
                if occ is not None and occ != foco:
                    rel.add(occ)
        if not rel:
            rel = {x for x, o in s.x.items() if o is not None}
        for x in rng.sample(sorted(rel), min(k, len(rel))):
            s.remove(x)
        s.log.clear()
        constructivo(ins, alpha, rng, sol=s)
        busqueda_local(s, t_lim=2.0)
        if s.objetivo() >= cur.objetivo():
            cur = s
            if s.objetivo() > best.objetivo():
                best = s.copia()
    return best


def busqueda_local(sol, prof_max=2, t_lim=30.0):
    """VND con primera mejora: N1 inserción directa, N2 expulsión prof. 1, N3 expulsión prof. 2,
    N4 sacar 1 – meter 2, N5 reubicación por zona, N6 swap de salas. Al mejorar se reinicia en N1."""
    vecindarios = [lambda s: N_insercion(s, 0)]
    vecindarios += [lambda s, p=p: N_insercion(s, p) for p in range(1, prof_max + 1)]
    vecindarios += [N_sacar1_meter2, N_reubicar, N_swap_salas]
    t0 = time.time(); j = 0
    while j < len(vecindarios) and time.time() - t0 < t_lim:
        if vecindarios[j](sol):
            sol.log.clear(); j = 0
        else:
            j += 1
    sol.log.clear()
    return sol


def grasp(ins, iters=20, alpha=0.3, seed=1, t_lim=60.0):
    rng = random.Random(seed)
    mejor = busqueda_local(constructivo(ins))          # iteración 0: voraz puro
    t0 = time.time()
    for _ in range(iters):
        if time.time() - t0 > t_lim or mejor.n_prog() == len(ins.S) and mejor.dispersion() == 0:
            break
        s = busqueda_local(constructivo(ins, alpha, rng))
        if s.objetivo() > mejor.objetivo():
            mejor = s
    return mejor


# ----------------------------------------------------------------------------- utilidades
def cota_superior(ins):
    """Cota: cirugías con ≥1 opción factible, acotada por capacidad de salas (|R|·|Days|·2)."""
    vacia = Solucion(ins)
    n = sum(1 for s in ins.S if vacia.opciones(s))
    return min(n, len(ins.R) * len(ins.Days) * len(ins.Blocks))


def imprimir(sol):
    ins = sol.ins
    print(f"{'Cirugía':8}{'Tipo':9}{'Día':5}{'Bloq':5}{'Sala':6}{'Zona':6}{'Cirujano':10}Dep")
    for s in ins.S:
        o = sol.x[s]
        if o:
            d, b, r, k = o
            print(f"{s:8}{ins.type_of[s]:9}{d:5}{b:5}{r:6}{ins.zone_of[r]:6}{k:10}{ins.dept_of.get(k)}")
    sin = [s for s in ins.S if sol.x[s] is None]
    print("Sin programar:", sin or "ninguna")


def generar(n_s=30, n_r=6, n_k=10, seed=0):
    """Instancia sintética con la misma estructura del enunciado (para pruebas)."""
    g = random.Random(seed)
    Days = ["L", "Ma", "Mi", "J", "V"]; Blocks = ["AM", "PM"]
    Types = ["Cardio", "Orto", "Neuro", "General"]; Zones = ["Z0", "Z1"]
    Deps = ["Dep0", "Dep1", "Dep2"]
    S = [f"C{i}" for i in range(n_s)]; R = [f"R{i}" for i in range(n_r)]
    K = [f"S{i}" for i in range(n_k)]
    rooms_type = {t: g.sample(R, g.randint(2, max(2, n_r // 2))) for t in Types}
    type_k = {k: g.sample(Types, g.randint(1, 2)) for k in K}
    for t in Types:
        if not any(t in v for v in type_k.values()):
            type_k[g.choice(K)].append(t)
    dept = {d: [] for d in Deps}
    for k in K:
        dept[g.choice(Deps)].append(k)
    return dict(Surgeries=S, Rooms=R, Days=Days, Blocks=Blocks, Types=Types,
                Departments=Deps, Zones=Zones, Surgeons=K,
                Type_Surgery={s: g.choice(Types) for s in S},
                Rooms_Type=rooms_type, Rooms_Z={r: g.choice(Zones) for r in R},
                Type_Surgeon=type_k, Dept_Surgeon=dept,
                Days_Surgery={s: sorted(g.sample(Days, g.randint(1, 3)), key=Days.index) for s in S},
                Days_Surgeon={k: sorted(g.sample(Days, g.randint(2, 4)), key=Days.index) for k in K})


def correr(D, nombre, args, detalle=False):
    ins = Instancia(D)
    t = time.time(); c = constructivo(ins); tc = time.time() - t
    assert not validar(c), validar(c)
    t = time.time(); ls = busqueda_local(c.copia()); tl = time.time() - t
    assert not validar(ls), validar(ls)
    t = time.time(); gr = grasp(ins, args.grasp, args.alpha, args.seed); tg = time.time() - t
    assert not validar(gr), validar(gr)
    ub = cota_superior(ins)
    print(f"{nombre:10} |S|={len(ins.S):3} UB={ub:3} | Constr {c.n_prog():3} disp {c.dispersion()} ({tc:.2f}s)"
          f" | +LS {ls.n_prog():3} disp {ls.dispersion()} ({tl:.2f}s)"
          f" | GRASP+LS {gr.n_prog():3} disp {gr.dispersion()} ({tg:.1f}s)")
    if detalle:
        imprimir(gr)
    return gr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instancias", nargs="*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--grasp", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--detalle", action="store_true")
    a = ap.parse_args()
    if a.demo:
        for i, (ns, nr, nk) in enumerate([(30, 6, 10), (40, 5, 10), (60, 6, 12), (80, 6, 14), (100, 8, 16)]):
            correr(generar(ns, nr, nk, seed=i), f"sint{i}", a, a.detalle and i == 0)
    for p in a.instancias:
        correr(cargar_instancia(p), p.split("/")[-1], a, a.detalle)
