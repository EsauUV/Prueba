from vpython import *
import time
import math
import copy
import numpy as np

dims = {"width": 640, "height": 480}
ventana = canvas(width = dims["width"], height = dims["height"])
ventana.range = 2000

default = vec(0, 0, 23000)
camara = {'obj': ventana.camera, 'pos': default}

G = 6.674e-11
delta_t = 0.0005
calc_p_seg = 2000
delta_calc = 1 / calc_p_seg
seleccion = None
objetivo = None
masa_input = None
radio_input = None
btn_nueva = None
nombre_input = None
input_nueva_masa = None
boton_pausa = None
play = True
now = 0; then = 0; offset = 0; _now = 0
a = None
b = None
e = None
T = None

posicion_inputs = {}
velocidad_inputs = {}

input_orbita = None

movimientos = []

def seleccionar(seleccion):
    global objetivo

    objetivo = seleccion
    radi = objetivo["obj"].radius
    pos_cam = vec(objetivo["pos"].x, objetivo["pos"].y, camara['pos'].z)

    masa_input.text = objetivo["masa"]
    radio_input.text = objetivo["rad"]
    nombre_input.text = objetivo["esf-eth"].text
    input_nueva_masa.checked = False

    _orbita = ""
    if "inf_orbita" in objetivo:
        _orbita = objetivo["inf_orbita"]["_orbita"]["esf-eth"].text
        
    input_orbita.text = _orbita

    if "puntos" in objetivo and "elipse" in objetivo["puntos"]:
        T.text = f"{objetivo['puntos']['elipse'][0]:.4f}"
        a.text = f"{objetivo['puntos']['elipse'][1]:.2f}"
        b.text = f"{objetivo['puntos']['elipse'][2]:.2f}"
        e.text = f"{objetivo['puntos']['elipse'][3]:.4f}"
    else:
        T.text = a.text = b.text = e.text = ""

    acualizar_vec()
            
    camara["pos"] = pos_cam
    
def acercar(ev):
    global objetivo

    if seleccion != None:
        seleccionar(seleccion)

ventana.bind('click', acercar)
        
def deslizante(s):
    global delta_t
    delta_t = 0.0005 * s.value

sli = slider(length = dims["width"] / 2, top = 15, bind = deslizante)
sli.value = 1
ventana.append_to_caption("Velocidad de la simulación\n")

def pausar(evt):
    global play, now, then, offset

    if play:
        then = now
        boton_pausa.text = "Reanudar animación"
    else:
        offset = _now - then
        boton_pausa.text = "Pausar animación"
    
    play = not play

velocidad_cam = 0
def controles(ev):
    global velocidad_cam
    
    if ev.which == 81 or 87:
        velocidad_cam += 1
        pos_cam = camara['pos']
        
        if ev.which == 81 and pos_cam.z > 10 * velocidad_cam:
            camara["pos"] += vec(0, 0, -velocidad_cam)
        elif ev.which == 87:
            camara["pos"] += vec(0, 0, velocidad_cam)


def frenar(ev):
    global velocidad_cam
    
    if ev.which == 81 or 87:
        velocidad_cam = 0

ventana.bind('keydown', controles)
ventana.bind('keyup', frenar)

def mover_camara():
    if objetivo != None:
        camara['pos'] = vec(objetivo['pos'].x, objetivo['pos'].y, camara['pos'].z)

def crear_masa(m, r, vec_pos, vec_vel, _label, _orbita):
    masa = sphere(radius = r, color = color.yellow, pos=vec_pos)
    esf_eth = label(text = _label, xoffset = 20, yoffset = 50, height = 16, font = 'courier')

    masa_meta = {"obj": masa, "masa": m, "rad": r, "pos": vec_pos, "vel": vec_vel, "esf-eth": esf_eth, "lista_pos": []}

    if _orbita != None:
        masa_meta["inf_orbita"] = iniciar_inf_orbita(masa_meta, _orbita)
    
    return masa_meta

tiempo = 0
def orbitas(now, prev_calc, delta_calc, escenas_pendientes, acumulado, masas):
    global tiempo
    
    if (now - prev_calc < delta_calc):
        return (prev_calc, escenas_pendientes, acumulado)

    tiempo += delta_t
    
    if (escenas_pendientes == 0):
        cociente = (now - prev_calc) / delta_calc
        escenas_pendientes = math.trunc(cociente)
        acumulado += cociente - escenas_pendientes

        if acumulado > 1:
            escenas_pendientes += 1
            acumulado -= 1
        prev_calc = now

    prev_pos = masas[1]["pos"]
    actualizadas = []
    
    for masa in masas:
        acc_res = vec(0, 0, 0)
        
        pos_act = masa["pos"]
        for influencia in masas:

            if masa["obj"] == influencia["obj"]:
                continue
  
            pos_inf = influencia["pos"]
            dir_act = pos_inf - pos_act
            d_cuadrada = dir_act.mag2
            acc = G * influencia["masa"] / d_cuadrada
            acc_res += acc * dir_act.norm()

        copia = copy.copy(masa)
        copia["vel"] += acc_res * delta_t
        copia["pos"] += copia["vel"] * delta_t
        actualizadas.append(copia)

    for masa, actual in zip(masas, actualizadas):
        actual_inf_orbita(masa, actual, tiempo)
        masa.update(actual)

    if (escenas_pendientes > 0):
        escenas_pendientes -= 1

    return (prev_calc, escenas_pendientes, acumulado)

def rayo_label(masa, rho):
    ray = ventana.mouse.ray
    forw = ventana.forward
    pos = ventana.camera.pos
    
    t = forw.mag2 / forw.dot(ray) * rho
    p = pos + t * ray

    lab = masa["esf-eth"]

    xoffset = lab.xoffset / 8
    yoffset = lab.yoffset / 8
    
    width = (len(lab.text) * lab.height * 0.52 + lab.border * 2) / 8
    height = (lab.height + lab.border * 2) / 8

    org = lab.pos + vec(xoffset - width / 2, yoffset, 0)
    
    if org.x < p.x < org.x + width and org.y < p.y < org.y + height:
        lab.color = color.blue
        return masa

    lab.color = color.white

    return None      

def obtener_punto(masa, rho):
    forw = ventana.forward
    pos = camara["pos"]
    rect = (masa["pos"] - pos).norm()
    t = forw.mag2 / forw.dot(rect) * rho

    #Posición del punto en donde se proyecta en el plano
    #imaginario que está a 50 metros frente a la cámara
    return pos + t * rect

def iniciar_inf_orbita(masa, _orbita):
    i = (masa["pos"] - _orbita["pos"]).norm()
    vxi = (masa['vel'] - _orbita["vel"]).cross(i).norm()

    return {"i": i, "vxi": vxi, "_orbita": _orbita, "prev_rxi_dot_vxi": None, "tiempo": None, "periodo": None}



def actual_inf_orbita(prev, actual, now):
    if not "inf_orbita" in actual:
        return
    
    inf = actual["inf_orbita"]
    i = inf["i"]
    vxi = inf["vxi"]
    _orbita = inf["_orbita"]

    r = actual["pos"] - _orbita["pos"]

    if (r.dot(i) <= 0 or r.cross(i).dot(vxi) < 0):
        inf["prev_rxi_dot_vxi"] = r.cross(i).dot(vxi)
    elif inf["prev_rxi_dot_vxi"] != None and r.dot(i) > 0 and inf["prev_rxi_dot_vxi"] <= 0 and r.cross(i).dot(vxi) >= 0:

        if inf["tiempo"] != None:
            periodo = now - inf['tiempo']
            
            iniciar_elipse(actual, periodo, now)

        inf["tiempo"] = now
        inf["prev_rxi_dot_vxi"] = None

    if "puntos" in inf and len(inf["puntos"]["arreglo"]) < 5:
        puntos_orbita(actual, now)

def iniciar_elipse(actual, periodo, now):
    global a, b, e, T
    
    inf = actual["inf_orbita"]
    _orbita = inf["_orbita"]
    x = (actual["pos"] - _orbita["pos"]).norm()
    v = actual["vel"]
    y = x.cross(v).cross(x).norm()

    inf_puntos = {}
    if "puntos" in inf and len(inf["puntos"]["arreglo"]) == 5:
        (_a, _b, _e) = ecuacion_elipse(inf['puntos']['arreglo'])
        inf_puntos["elipse"] = (periodo, _a, _b, _e)

        if actual['esf-eth'].text == objetivo['esf-eth'].text:
            a.text = f"{_a:.2f}"
            b.text = f"{_b:.2f}"
            e.text = f"{_e:.4f}"
            T.text = f"{periodo:.4f}"

    inf["puntos"] = inf_puntos
    inf["puntos"]["x"] = x
    inf["puntos"]["y"] = y
    inf["puntos"]["periodo"] = periodo / 6
    inf["puntos"]["pasado"] = 0
    inf["puntos"]["arreglo"] = []

def puntos_orbita(actual, now):
    inf = actual["inf_orbita"]

    #print(f"Los puntos son {now - inf['puntos']['pasado']} {now - inf['puntos']['pasado'] < inf['puntos']['periodo']}")

    if now - inf["puntos"]["pasado"] < inf["puntos"]["periodo"]:
        return

    inf["puntos"]["pasado"] = now
    _orbita = inf["_orbita"]
    r = actual["pos"] - _orbita["pos"]
    vec_coords = (r.dot(inf["puntos"]["x"]), r.dot(inf["puntos"]["y"]))
    inf["puntos"]["arreglo"].append(vec_coords)

def ecuacion_elipse(arreglo):
    f = lambda xy: [xy[0] ** 2, xy[0] * xy[1], xy[1] ** 2, xy[0], xy[1]]
    
    A = np.array(list(map(f, arreglo)))
    b = np.array([-1] * len(A))
    X = np.linalg.solve(A, b)

    u = (X[2] - X[0]) / X[1]
    m = u + math.sqrt(u ** 2 + 1)

    B2_4AC = 4 * X[0] * X[2] - X[1] ** 2
    h = (X[4] * X[1] - 2 * X[2] * X[3]) / B2_4AC
    k = (X[1] * X[3] - 2 * X[0] * X[4]) / B2_4AC

    _F = 1
    for (coef, val) in zip(f((h, k)), X):
        _F += coef * val

    _F = abs(_F * (m ** 2 + 1))
    _A_F = _F / abs(X[0] + X[1] * m + X[2] * m ** 2)
    _B_F = _F / abs(X[0] * m ** 2 - X[1] * m + X[2])
    
    a = max(_A_F, _B_F) ** 0.5
    b = min(_A_F, _B_F) ** 0.5
    e = (1 - b ** 2 / a ** 2) ** 0.5

    return (a, b, e)

def rotar_vector(v, u, theta):
    alpha = v.dot(u) / u.dot(u)
    w = v - alpha * u    
    s = w.cross(u)
    p = w.mag / s.mag * s
    theta_rad = theta * math.pi / 180
    _w = math.cos(theta_rad) * w + math.sin(theta_rad) * p

    return _w + alpha * u

def parse_float(text):
    try:
        return float(text)
    except:
        print("Algo salió mal...")

    return None

def input_masa(ev):
    masa = parse_float(ev.text) or objetivo["masa"]

    if objetivo != None:
        objetivo["masa"] = masa

def input_radio(ev):
    rad = parse_float(ev.text) or objetivo["rad"]

    if objetivo != None:
        objetivo["rad"] = rad
        objetivo["obj"].radius = rad

def input_nombre(ev):
    if objetivo != None and ev.text and len(ev.text):
        objetivo["esf-eth"].text = ev.text
        
    return

def input_vec(ev, clave, coord):
    pos = parse_float(ev.text)

    if objetivo != None:
        if coord == "x":
            objetivo[clave].x = pos or objetivo[clave].x
        elif coord == "y":
            objetivo[clave].y = pos or objetivo[clave].y
        elif coord == "z":
            objetivo[clave].z = pos or objetivo[clave].z
            
def nueva_masa(masas):
    m = parse_float(masa_input.text)
    r = parse_float(radio_input.text)

    vec_pos = None
    vec_vel = None
    try:
        px = parse_float(posicion_inputs["x"].text)
        py = parse_float(posicion_inputs["y"].text)
        pz = parse_float(posicion_inputs["z"].text)
        
        vx = parse_float(velocidad_inputs["x"].text)
        vy = parse_float(velocidad_inputs["y"].text)
        vz = parse_float(velocidad_inputs["z"].text)

        vec_pos = vec(px, py, pz)
        vec_vel = vec(vx, vy, vz)
    except:
        print("Error al leer los vectores")
        return

    _label =   nombre_input.text

    if _label in map(lambda masa: masa['esf-eth'].text, masas):
        print(f"Masa repetida {_label}")
        return

    if input_orbita.text and len(input_orbita.text) > 0:
        _orbita = next(filter(lambda masa : masa["esf-eth"].text == input_orbita.text, masas))

        if not _orbita:
            print("No existe el cuerpo a orbitar")
            return

    if None in (m, r, _label):
        print("Hay campos obligatorios vacíos")
        return
    
    m_nueva = crear_masa(m, r, vec_pos, vec_vel, _label, _orbita)

    masas.append(m_nueva)
    
    return

def nueva_masa_list(ev):
    btn_nueva.disabled = not input_nueva_masa.checked

def verificar_check(*args):
    if (input_nueva_masa.checked):
        return

    argum = [args[0]]
    for i in range(2, len(args)):
        argum.append(args[i])

    args[1](*argum)
    

def iniciar_interfaz(masas):
    global m, masa_input, radio_input, nombre_input, input_orbita, input_nueva_masa, btn_nueva, boton_pausa
    global a, b, e, T
    
    ventana.append_to_caption("\n")
    boton_pausa = button(text = "Pausar animación", bind = pausar)
    ventana.append_to_caption("\n")
    ventana.append_to_caption("\nCuerpo: ")
    nombre_input = winput(text = f"{masas[0]['esf-eth'].text}", bind = lambda ev : verificar_check(ev, input_nombre))
    ventana.append_to_caption(" Masa: ")
    masa_input = winput(text = f"{masas[0]['masa']}", bind = lambda ev : verificar_check(ev, input_masa))
    ventana.append_to_caption(" Radio: ")
    radio_input = winput(text = f"{masas[0]['rad']}", bind = lambda ev : verificar_check(ev, input_radio))

    ventana.append_to_caption("\n\nPeriodo: ")
    T = winput(bind = lambda : None)
    ventana.append_to_caption(" Semieje mayor: ")
    a = winput(bind = lambda : None)
    ventana.append_to_caption(" Semieje menor: ")
    b = winput(bind = lambda : None)
    ventana.append_to_caption("\n\nExcentricidad: ")
    e = winput(bind = lambda : None)

    
    ventana.append_to_caption("\n\nPosición x: ")
    posicion_inputs["x"] = winput(text = f"{masas[0]['pos'].x:.2f}", bind = lambda ev: verificar_check(ev, input_vec, "pos", "x"))
    ventana.append_to_caption(" y: ")
    posicion_inputs["y"] = winput(text = f"{masas[0]['pos'].y:.2f}", bind = lambda ev: verificar_check(ev, input_vec, "pos", "y"))
    ventana.append_to_caption(" z: ")
    posicion_inputs["z"] = winput(text = f"{masas[0]['pos'].z:.2f}", bind = lambda ev: verificar_check(ev, input_vec, "pos", "z"))
    ventana.append_to_caption("\n\nVelocidad x: ")
    velocidad_inputs["x"] = winput(text = f"{masas[0]['vel'].x:.2f}", bind = lambda ev: verificar_check(ev, input_vec, "vel", "x"))
    ventana.append_to_caption(" y: ")
    velocidad_inputs["y"] = winput(text = f"{masas[0]['vel'].y:.2f}", bind = lambda ev: verificar_check(ev, input_vec, "vel", "y"))
    ventana.append_to_caption(" z: ")
    velocidad_inputs["z"] = winput(text = f"{masas[0]['vel'].z:.2f}", bind = lambda ev: verificar_check(ev, input_vec, "vel", "z"))
    ventana.append_to_caption("\n\nOrbita a : ")
    input_orbita = winput(text = "", bind = lambda : None)
    ventana.append_to_caption("\n\n")
    
    input_nueva_masa = checkbox(bind = nueva_masa_list, text='Nueva masa')

    ventana.append_to_caption("\n\n")
    btn_nueva = button(text = "Crear", bind = lambda ev: nueva_masa(masas), disabled = True)

def acualizar_vec():
    if objetivo == None or input_nueva_masa.checked:
        return

    posicion_inputs["x"].text = f"{objetivo['pos'].x:.2f}"
    posicion_inputs["y"].text = f"{objetivo['pos'].y:.2f}"
    posicion_inputs["z"].text = f"{objetivo['pos'].z:.2f}"
    velocidad_inputs["x"].text = f"{objetivo['vel'].x:.2f}"
    velocidad_inputs["y"].text = f"{objetivo['vel'].y:.2f}"
    velocidad_inputs["z"].text = f"{objetivo['vel'].z:.2f}"

def trayectoria(masa, posicion_plano):
    posiciones_masa = masa["lista_pos"]
    l = len(posiciones_masa)

    ult_cuadro = None
    if l == 100:
        ult = posiciones_masa.pop(0)

        if "cuadro" in  ult:
            ult_cuadro = ult["cuadro"]
    
    nueva = None
    if l > 0:
        ultima = posiciones_masa[-1]
        p = ultima["pos"]

        di = (posicion_plano - p).norm() * 0.1
        dip = vec(-di.y, di.x, di.z).norm() * 0.1

        #creando los vértices de los cuadros para dibujar la
        #trayectoria

        if "cuadro" in ultima:
            ultima["cuadro"].visible = False

        q = None
        if ult_cuadro:
            ult_cuadro.visible = False
            ult_cuadro.vs[0].pos = p - di + dip
            ult_cuadro.vs[1].pos = p - di - di
            ult_cuadro.vs[2].pos = posicion_plano + di - dip
            ult_cuadro.vs[3].pos = posicion_plano + di + dip
            q = ult_cuadro
        else:
            a = vertex(pos = p - di + dip)
            b = vertex(pos = p - di - dip)
            c = vertex(pos = posicion_plano + di - dip)
            d = vertex(pos = posicion_plano + di + dip)        
            q = quad(vs = [a, b, c, d], visible = False)
        
        nueva = {"pos": posicion_plano, "cuadro": q}
    else:
        nueva = {"pos": posicion_plano}
    
    masa["lista_pos"].append(nueva)

def regla():
    b1 = box(pos = vec(0, 0, 0), length = 0.3, height = 2, width = 0.1)
    b2 = box(pos = vec(0, 0, 0), length = 10.0, height = 0.3, width = 0.1)
    b3 = box(pos = vec(0, 0, 0), length = 0.3, height = 2, width = 0.1)

    return (b1, b2, b3)

def calc_dist(escala, rho, boxes):
    p_cam = camara["pos"]
    (b1, b2, b3) = boxes
    
    p1 = p_cam + vec(25, -28, - rho - 1)
    p2 = p_cam + vec(35, -28, - rho - 1)

    b1.pos = p1
    b2.pos = (p1 + p2) / 2
    b3.pos = p2

    u1 = p1 - p_cam
    t1 = - p_cam.z / u1.z
    _p1 = p_cam + t1 * u1

    u2 = p2 - p_cam
    t2 = - p_cam.z / u2.z
    _p2 = p_cam + t2 * u2

    escala.pos = b2.pos + vec(0, 2, 0)
    escala.text = f"{(_p2 - _p1).mag:.2f}[m]"

def main():
    global seleccion, objetivo, now, offset, _now
    
    pos_tierra = vec(15209.8, 0, 0)
    m1 = crear_masa(1.989e23, 69.63, vec(0, 0, 0), vec(0, 0, 0), "Sol", None)
    m2 = crear_masa(5.972e17, 0.64, pos_tierra, vec(0, 29294.7, 0), "Tierra", m1)

    
    
    m3 = crear_masa(7.349e15, 0.17, pos_tierra + vec(40.5, 0, 0), vec(0, 30336.5, 0), "Luna", m2)

    pos_venus = vec(9064.4, 6042.9, 0)
    vel_venus = vec(-19297.7, 28946.6, 0)
    rotado = rotar_vector(vel_venus, pos_venus, 3.39471)
    #rotado = rotar_vector(vel_venus, pos_venus, 90)
    
    m4 = crear_masa(4.869e17, 0.61, pos_venus, rotado, "Venus", m1)

    pos_merc = vec(6981.68775, 0, 0)
    vel_merc = vec(0, 38863.4787, 0)
    m5 = crear_masa(3.30200e16, 0.24, pos_merc, vel_merc, "Mercurio", m1)

    masas = [m1, m2, m3, m4, m5]

    iniciar_interfaz(masas)

    c_p_seg = 30
    delta_fotog = 1 / c_p_seg

    now = prev = prev_h = prev_calc = time.time()
    escenas_pendientes = 0
    acumulado = 0

    rho = 50

    seleccion = None
    objetivo = m1
    dist = 1000000

    h_p_seg = 60
    delta_h = 1 / h_p_seg

    (b1, b2, b3) = regla()
    escala = label(text = "0", xoffset = 0, yoffset = 0, height = 12, font = 'courier', box = False)
    
    while True:        
        _now = time.time()


        if play:
            now = _now - offset
            (prev_calc, escenas_pendientes, acumulado) = orbitas(now, prev_calc, delta_calc, escenas_pendientes, acumulado, masas)
            mover_camara()
        
        if (_now - prev > delta_fotog):
            prev = _now

            #print(f"Distancia menor {dist}")
            _seleccion = None
            for masa in masas:
                masa["obj"].pos = masa["pos"]
                _seleccion = rayo_label(masa, rho) or _seleccion
                posicion_plano = obtener_punto(masa, rho)
                masa["esf-eth"].pos = posicion_plano
                
            seleccion = _seleccion

            if play:
                acualizar_vec()

            calc_dist(escala, rho, (b1, b2, b3))
            
            ventana.camera.pos = camara["pos"]

        if _now - prev_h > delta_h:
            prev_h = _now
        
            for masa in masas:
                posicion_plano = obtener_punto(masa, rho)
                trayectoria(masa, posicion_plano)
main()
