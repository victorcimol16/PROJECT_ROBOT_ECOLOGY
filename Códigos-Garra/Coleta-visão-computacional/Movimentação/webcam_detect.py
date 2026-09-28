import time

import cv2
from ultralytics import YOLO
from pymata4.pymata4 import Pymata4

# =========================
# CONFIGURACOES
# =========================
MODEL_PATH = r"C:\Users\Acer\Desktop\desafio projeto\dataset\runs\detect\tampinha_teste_novo7\weights\best.pt"
PORTA = 'COM5'
CAMERA = 0
CONF = 0.45
FRAME_SKIP = 3  # roda a deteccao a cada N frames (reaproveita a ultima deteccao nos demais)

PINOS = {"base": 11, "ombro": 10, "ante": 9, "garra": 6}

# angulos de repouso / trabalho de cada servo
ANGULO_BASE_INICIAL = 90
ANGULO_BASE_DESCARTE = 20     # posicao da base onde o objeto e solto
ANGULO_OMBRO_REPOUSO = 90
ANGULO_ANTE_REPOUSO = 90       # invertido: 180 - 90
ANGULO_OMBRO_BAIXO = 160
ANGULO_ANTE_BAIXO = 40         # invertido: 180 - 180
BASE_MIN, BASE_MAX = 0, 180
PASSO_BASE = 2

GARRA_ABERTA = 110
GARRA_FECHADA = 40

PASSOS_MOVIMENTO = 50   # granularidade das rampas ombro/antebraco
ATRASO_PASSO = 0.02     # tempo entre cada grau de uma rampa

# controle de busca (evita o servo da base ficar reagindo a ruido de deteccao)
INTERVALO_BUSCA = 0.2            # tempo minimo entre comandos da base
CONFIRMACOES_NECESSARIAS = 3     # frames seguidos concordando antes de mover a base
MAX_REVERSOES = 4                # reversoes de sentido em pouco tempo = oscilacao (hunting)
JANELA_REVERSAO = 3.0
COOLDOWN_OSCILACAO = 1.5         # pausa pra deixar o servo descansar apos oscilar demais

# =========================
# ARDUINO / SERVOS (pymata4 + firmware FirmataExpress)
# =========================
board = Pymata4(com_port=PORTA)

angulo_atual = {
    "base": ANGULO_BASE_INICIAL,
    "ombro": ANGULO_OMBRO_REPOUSO,
    "ante": ANGULO_ANTE_REPOUSO,
    "garra": GARRA_ABERTA,
}


def ligar(nome):
    """Reativa o servo (modo servo) na posicao em que ele estava."""
    board.set_pin_mode_servo(PINOS[nome])
    board.servo_write(PINOS[nome], angulo_atual[nome])


def desligar(nome):
    """Corta o PWM do servo - sem torque, sem aquecer/estressar parado."""
    board.set_pin_mode_digital_output(PINOS[nome])
    board.digital_write(PINOS[nome], 0)


def mover_suave(nome, alvo, atraso=ATRASO_PASSO):
    """Move um servo em rampa, um grau de cada vez, a partir da posicao atual."""
    pino = PINOS[nome]
    board.set_pin_mode_servo(pino)
    inicio = angulo_atual[nome]
    passo = 1 if alvo >= inicio else -1
    for ang in range(int(inicio), int(alvo) + passo, passo):
        board.servo_write(pino, ang)
        time.sleep(atraso)
    angulo_atual[nome] = alvo


def mover_dois_suave(nome1, alvo1, nome2, alvo2, passos=PASSOS_MOVIMENTO, atraso=ATRASO_PASSO):
    """Move dois servos juntos e de forma coordenada (ex.: ombro + antebraco)."""
    pino1, pino2 = PINOS[nome1], PINOS[nome2]
    board.set_pin_mode_servo(pino1)
    board.set_pin_mode_servo(pino2)
    inicio1, inicio2 = angulo_atual[nome1], angulo_atual[nome2]
    for i in range(passos + 1):
        fracao = i / passos
        board.servo_write(pino1, int(inicio1 + (alvo1 - inicio1) * fracao))
        board.servo_write(pino2, int(inicio2 + (alvo2 - inicio2) * fracao))
        time.sleep(atraso)
    angulo_atual[nome1] = alvo1
    angulo_atual[nome2] = alvo2


def ajustar_base(delta):
    """Pequeno ajuste incremental da base, respeitando os limites mecanicos."""
    novo = max(BASE_MIN, min(BASE_MAX, angulo_atual["base"] + delta))
    if novo != angulo_atual["base"]:
        board.set_pin_mode_servo(PINOS["base"])  # garante que esta ligada (pode ter sido desligada apos a coleta anterior)
        board.servo_write(PINOS["base"], novo)
        angulo_atual["base"] = novo


def abrir_garra():
    mover_suave("garra", GARRA_ABERTA)


def fechar_garra():
    mover_suave("garra", GARRA_FECHADA)


# posiciona tudo uma vez e desliga o que fica ocioso ate o primeiro ciclo de coleta
ligar("base")
ligar("ombro")
ligar("ante")
ligar("garra")
time.sleep(0.3)
desligar("ombro")
desligar("ante")
desligar("garra")

# =========================
# YOLO + CAMERA
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(
        f"Nao foi possivel abrir a camera de indice {CAMERA}. "
        "Tente outro valor para CAMERA (0, 1, 2...) ou verifique se a webcam esta conectada."
    )
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

cv2.namedWindow("ROBO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROBO", 640, 360)


def detectar_objeto(frame):
    """Roda o YOLO no frame e retorna a caixa (x1, y1, x2, y2) da 1a deteccao, ou None."""
    res = model.predict(frame, conf=CONF, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        return tuple(map(int, res.boxes[0].xyxy[0]))
    return None


def classificar_posicao(cx, limite_esq, limite_dir):
    if cx < limite_esq:
        return "ESQUERDA"
    if cx > limite_dir:
        return "DIREITA"
    return "MEIO"


# =========================
# CONTROLE DE BUSCA (com anti-oscilacao)
# =========================
estado_busca = {}


def resetar_busca():
    estado_busca.update({
        "pos_anterior": None,
        "contagem_pos": 0,
        "ultimo_mov": time.time(),
        "ultima_direcao": None,
        "contagem_reversoes": 0,
        "tempo_primeira_reversao": None,
        "em_cooldown": False,
        "fim_cooldown": 0,
    })


resetar_busca()


def processar_busca(pos):
    """
    Atualiza o servo da base com base na posicao detectada.
    So move a base apos N deteccoes seguidas concordando (evita reagir a ruido)
    e entra em cooldown se ficar invertendo de sentido demais (evita hunting).
    Retorna True quando o objeto esta alinhado e pronto pra ser pego.
    """
    e = estado_busca
    agora = time.time()

    if pos == e["pos_anterior"]:
        e["contagem_pos"] += 1
    else:
        e["contagem_pos"] = 1
        e["pos_anterior"] = pos

    if e["em_cooldown"]:
        if agora >= e["fim_cooldown"]:
            e["em_cooldown"] = False
            ligar("base")
        return False

    if agora - e["ultimo_mov"] < INTERVALO_BUSCA or e["contagem_pos"] < CONFIRMACOES_NECESSARIAS:
        return False

    e["ultimo_mov"] = agora
    direcao = "E" if pos == "ESQUERDA" else ("D" if pos == "DIREITA" else None)

    if direcao and e["ultima_direcao"] and direcao != e["ultima_direcao"]:
        if e["tempo_primeira_reversao"] is None or agora - e["tempo_primeira_reversao"] > JANELA_REVERSAO:
            e["tempo_primeira_reversao"] = agora
            e["contagem_reversoes"] = 1
        else:
            e["contagem_reversoes"] += 1

    if e["contagem_reversoes"] >= MAX_REVERSOES:
        print(f"Muita oscilacao de sentido -> pausando {COOLDOWN_OSCILACAO}s pro servo descansar")
        e["em_cooldown"] = True
        e["fim_cooldown"] = agora + COOLDOWN_OSCILACAO
        e["contagem_reversoes"] = 0
        e["tempo_primeira_reversao"] = None
        desligar("base")
        return False

    if direcao == "E":
        ajustar_base(PASSO_BASE)
        e["ultima_direcao"] = "E"
    elif direcao == "D":
        ajustar_base(-PASSO_BASE)
        e["ultima_direcao"] = "D"
    elif pos == "MEIO":
        desligar("base")
        return True

    return False


# =========================
# SEQUENCIA DE COLETA
# =========================
def executar_coleta_e_descarte():
    """Desce, fecha a garra, sobe, leva ate o ponto de descarte e solta o objeto."""
    print("Alinhado! Pegando objeto...")
    ligar("ombro")
    ligar("ante")
    ligar("garra")

    abrir_garra()
    time.sleep(1)

    mover_dois_suave("ombro", ANGULO_OMBRO_BAIXO, "ante", ANGULO_ANTE_BAIXO)
    time.sleep(1)

    fechar_garra()
    time.sleep(1)

    # subida e contra o peso do braco+garra+objeto (mais exigente que descer, que tem
    # ajuda da gravidade) - vai mais devagar pra dar margem de torque ao servo
    mover_dois_suave("ombro", ANGULO_OMBRO_REPOUSO, "ante", ANGULO_ANTE_REPOUSO, atraso=0.04)
    print("Objeto pego! Levando ate o ponto de descarte...")

    ligar("base")
    mover_suave("base", ANGULO_BASE_DESCARTE)
    time.sleep(0.5)

    abrir_garra()
    time.sleep(0.5)

    # volta a base pro centro pra nao ficar no campo de visao da camera durante a busca
    mover_suave("base", ANGULO_BASE_INICIAL)
    time.sleep(0.3)  # da tempo do servo terminar de girar antes de cortar a energia

    for nome in ("base", "ombro", "ante", "garra"):
        desligar(nome)

    print("Objeto solto! Voltando a procurar...")


# =========================
# LOOP PRINCIPAL
# =========================
frame_count = 0
ultima_deteccao = None
modo = "procurando"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    altura, largura, _ = frame.shape

    zona_meio = int(largura * 0.2)
    centro = largura // 2
    limite_esq = centro - zona_meio // 2
    limite_dir = centro + zona_meio // 2

    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        ultima_deteccao = detectar_objeto(frame)

    annotated = frame.copy()
    pos = "SEM DETECCAO"

    if ultima_deteccao is not None:
        x1, y1, x2, y2 = ultima_deteccao
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cx = (x1 + x2) // 2
        pos = classificar_posicao(cx, limite_esq, limite_dir)

        cv2.putText(annotated, pos, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if modo == "procurando" and processar_busca(pos):
            modo = "pegando"
    else:
        print("Nada detectado")

    if modo == "pegando":
        executar_coleta_e_descarte()
        ultima_deteccao = None
        resetar_busca()
        modo = "procurando"

    cv2.putText(annotated, f"Modo: {modo}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(annotated, f"Base: {angulo_atual['base']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("ROBO", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# FINALIZACAO
# =========================
cap.release()
cv2.destroyAllWindows()
board.shutdown()
