import time
import traceback

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

# sensor ultrassonico HC-SR04 (mede a distancia ate o objeto pra calcular a descida)
PINO_TRIG = 12
PINO_ECHO = 13
DIST_MIN_CM, DIST_MAX_CM = 4, 40   # faixa util de leitura (fora disso = leitura descartada)
AMOSTRAS_DISTANCIA = 5             # numero de leituras pra tirar a mediana e filtrar ruido
DIST_MAX_COLETA = 14.0             # objeto acima disso e IGNORADO (nao entra em modo de coleta)
AJUSTE_COLETA_CM = 0.5             # desconto aplicado a distancia medida antes de calcular a descida

# aproximacao gradual da garra: em vez de ir direto pros angulos do objeto, comeca
# recuada e avanca ate a medida real (ex.: comeca em 10cm e desliza ate 12cm)
APROX_INICIO_CM = 2.0       # o quanto a garra comeca recuada antes da distancia do objeto
APROX_ATRASO_FINAL = 0.05   # atraso por passo no trecho final (recuado -> objeto); maior = mais lento
APROX_SUBPASSOS = 6         # sub-passos por grau no pouso final; maior = mais fino/suave (pousa em vez de cair)

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

# recuperacao de falha de comunicacao com o Arduino
TENTATIVAS_CONEXAO = 3           # tentativas por (re)conexao
ESPERA_RECONEXAO = 2.0           # segundos entre tentativas (da tempo do Arduino reiniciar)
MAX_FALHAS_SEGUIDAS = 3          # ciclos falhando seguido antes de desistir (provavel problema fisico)


# =========================
# CONEXAO COM O ARDUINO
# =========================
board = None  # sera preenchido por conectar_board(); pode ser recriado numa reconexao


def conectar_board(tentativas=TENTATIVAS_CONEXAO):
    """
    (Re)conecta no Arduino e configura o sonar, com algumas tentativas.
    Atualiza o global `board`. Retorna True/False.
    Captura ate SystemExit porque o pymata4 pode chamar sys.exit() ao falhar na serial.
    """
    global board
    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        try:
            board = Pymata4(com_port=PORTA)
            # sonar: dispara no TRIG e mede o retorno no ECHO. A leitura roda numa thread
            # de fundo do pymata4; sonar_read() devolve sempre a ultima medida em cache.
            board.set_pin_mode_sonar(PINO_TRIG, PINO_ECHO)
            time.sleep(0.2)  # da tempo do primeiro ciclo de leitura popular o cache
            return True
        except (Exception, SystemExit) as err:
            ultimo_erro = err
            print(f"[conexao] tentativa {tentativa}/{tentativas} em {PORTA} falhou: {err!r}")
            board = None
            time.sleep(ESPERA_RECONEXAO)
    print(f"[conexao] nao consegui conectar em {PORTA}. Ultimo erro: {ultimo_erro!r}")
    return False


angulo_atual = {
    "base": ANGULO_BASE_INICIAL,
    "ombro": ANGULO_OMBRO_REPOUSO,
    "ante": ANGULO_ANTE_REPOUSO,
    "garra": GARRA_ABERTA,
}


# =========================
# CONTROLE DOS SERVOS
# =========================
def ligar(nome):
    """Reativa o servo (modo servo) na posicao em que ele estava."""
    board.set_pin_mode_servo(PINOS[nome])
    board.servo_write(PINOS[nome], angulo_atual[nome])


def desligar(nome):
    """Corta o PWM do servo - sem torque, sem aquecer/estressar parado."""
    board.set_pin_mode_digital_output(PINOS[nome])
    board.digital_write(PINOS[nome], 0)


def desligar_todos():
    """Corta o PWM de todos os servos (usado na finalizacao e na recuperacao)."""
    for nome in ("base", "ombro", "ante", "garra"):
        try:
            desligar(nome)
        except (Exception, SystemExit):
            pass  # se a placa ja caiu, nao adianta insistir


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


def mover_dois_suave(nome1, alvo1, nome2, alvo2, passos=None, atraso=ATRASO_PASSO):
    """
    Move dois servos juntos e de forma coordenada (ex.: ombro + antebraco).
    Se passos=None, usa rampa proporcional: ~1 passo por grau do servo que se move
    mais - evita escrever dezenas de vezes a toa em movimentos pequenos (menos
    trafego na serial), mantendo a suavidade em movimentos grandes.
    """
    pino1, pino2 = PINOS[nome1], PINOS[nome2]
    board.set_pin_mode_servo(pino1)
    board.set_pin_mode_servo(pino2)
    inicio1, inicio2 = angulo_atual[nome1], angulo_atual[nome2]
    if passos is None:
        passos = max(1, int(round(max(abs(alvo1 - inicio1), abs(alvo2 - inicio2)))))
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


def posicionar_inicial():
    """Coloca os servos numa postura conhecida e desliga os ociosos ate a 1a coleta."""
    angulo_atual.update({
        "base": ANGULO_BASE_INICIAL,
        "ombro": ANGULO_OMBRO_REPOUSO,
        "ante": ANGULO_ANTE_REPOUSO,
        "garra": GARRA_ABERTA,
    })
    ligar("base")
    ligar("ombro")
    ligar("ante")
    ligar("garra")
    time.sleep(0.3)
    desligar("ombro")
    desligar("ante")
    desligar("garra")
    # a base fica ligada durante a busca


def recuperar_board():
    """
    Chamado quando a comunicacao com o Arduino cai no meio da operacao.
    Fecha a conexao antiga, reconecta e reposiciona. Retorna True se recuperou.
    """
    global board
    try:
        board.shutdown()
    except (Exception, SystemExit):
        pass
    time.sleep(1.0)
    if not conectar_board():
        return False
    try:
        posicionar_inicial()
    except (Exception, SystemExit):
        return False
    return True


# =========================
# SENSOR / CALCULO DE ANGULOS
# =========================
def ler_distancia():
    """
    Le o sensor ultrassonico varias vezes e retorna a mediana em cm,
    descartando leituras fora da faixa util. Retorna None se nenhuma leitura
    valida (objeto fora de alcance / sensor sem eco).
    """
    amostras = []
    for _ in range(AMOSTRAS_DISTANCIA):
        try:
            leitura = board.sonar_read(PINO_TRIG)
        except Exception:
            leitura = None
        # sonar_read pode devolver [dist, ts], None ou lista vazia dependendo da versao/estado
        valor = leitura[0] if leitura else None
        if valor and DIST_MIN_CM <= valor <= DIST_MAX_CM:
            amostras.append(valor)
        time.sleep(0.05)
    if not amostras:
        return None
    amostras.sort()
    return amostras[len(amostras) // 2]


def angulos_por_formula(alcance_cm):
    """Converte a distancia medida (cm) nos angulos de antebraco e ombro da coleta."""
    ombro = 126.1 + 3.80 * alcance_cm
    ante = -32.3 + 8.13 * alcance_cm
    return round(ante), round(ombro)


def _clamp180(ang):
    """Mantem o angulo dentro do curso mecanico do servo (evita forcar o batente)."""
    return max(0, min(180, ang))


def descer_ate_objeto(distancia_alvo, atraso=0.03):
    """
    Desce ate o objeto em duas etapas:
      1) rampa proporcional ate a postura recuada (APROX_INICIO_CM antes do alvo);
      2) POUSO FINAL lento e fino (recuado -> objeto) pra encostar de leve, nao cair em cima.
    Como a formula e linear na distancia, a 2a etapa equivale a percorrer 10 -> 12 cm.
    """
    inicio_cm = max(DIST_MIN_CM, distancia_alvo - APROX_INICIO_CM)

    ante_rec, ombro_rec = angulos_por_formula(inicio_cm)
    ante_alvo, ombro_alvo = angulos_por_formula(distancia_alvo)

    ante_rec, ombro_rec = _clamp180(ante_rec), _clamp180(ombro_rec)
    ante_alvo, ombro_alvo = _clamp180(ante_alvo), _clamp180(ombro_alvo)

    # 1) desce ate a postura recuada (rampa proporcional, rapida o suficiente)
    mover_dois_suave("ombro", ombro_rec, "ante", ante_rec, atraso=atraso)

    # 2) pouso final: muitos sub-passos (APROX_SUBPASSOS por grau), devagar
    graus = max(abs(ombro_alvo - ombro_rec), abs(ante_alvo - ante_rec))
    passos_final = max(1, int(round(graus * APROX_SUBPASSOS)))
    mover_dois_suave("ombro", ombro_alvo, "ante", ante_alvo,
                     passos=passos_final, atraso=APROX_ATRASO_FINAL)


# =========================
# VISAO (YOLO)
# =========================
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
def executar_coleta_e_descarte(distancia):
    """
    Desce, fecha a garra, sobe, leva ate o ponto de descarte e solta o objeto.
    `distancia` e a distancia (cm) ja medida no alinhamento. A descida usa essa
    distancia com um desconto de AJUSTE_COLETA_CM.
    """
    print("Alinhado! Pegando objeto...")
    ligar("ombro")
    ligar("ante")
    ligar("garra")

    abrir_garra()
    time.sleep(1)

    # aplica o desconto e desce se aproximando de pouco em pouco ate o objeto
    alvo_cm = max(DIST_MIN_CM, distancia - AJUSTE_COLETA_CM)
    print(f"Distancia: {distancia:.1f} cm (ajustada p/ {alvo_cm:.1f} cm) -> aproximando gradualmente")
    descer_ate_objeto(alvo_cm)
    time.sleep(1)

    fechar_garra()
    time.sleep(1)

    # subida: contra o peso do braco+garra+objeto (mais exigente que descer) - vai mais
    # devagar pra dar margem de torque ao servo e reduzir o pico de corrente
    mover_dois_suave("ombro", ANGULO_OMBRO_REPOUSO, "ante", ANGULO_ANTE_REPOUSO, atraso=0.04)
    time.sleep(0.3)
    print("Objeto pego! Levando ate o ponto de descarte...")

    ligar("base")
    mover_suave("base", ANGULO_BASE_DESCARTE)
    time.sleep(0.5)

    abrir_garra()
    time.sleep(0.5)

    # volta a base pro centro pra nao ficar no campo de visao da camera durante a busca
    mover_suave("base", ANGULO_BASE_INICIAL)
    time.sleep(0.3)  # da tempo do servo terminar de girar antes de cortar a energia

    desligar_todos()
    print("Objeto solto! Voltando a procurar...")


# =========================
# INICIALIZACAO
# =========================
if not conectar_board():
    input(
        f"\nFalha ao conectar no Arduino em {PORTA}.\n"
        "Confira a porta COM, se o FirmataExpress esta gravado e se nada mais usa a porta.\n"
        "Pressione ENTER para fechar..."
    )
    raise SystemExit(1)

print("Carregando modelo YOLO...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
if not cap.isOpened():
    try:
        board.shutdown()
    except (Exception, SystemExit):
        pass
    input(
        f"\nNao foi possivel abrir a camera de indice {CAMERA}.\n"
        "Tente outro valor para CAMERA (0, 1, 2...) ou verifique se a webcam esta conectada.\n"
        "Pressione ENTER para fechar..."
    )
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

cv2.namedWindow("ROBO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROBO", 640, 360)

posicionar_inicial()
resetar_busca()

# =========================
# LOOP PRINCIPAL
# =========================
frame_count = 0
ultima_deteccao = None
modo = "procurando"
distancia_coleta = None  # distancia medida no alinhamento, usada pela coleta
falhas_seguidas = 0  # ciclos com falha de placa seguidos (sem uma coleta bem-sucedida no meio)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler frame da camera -> encerrando.")
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

        # --- tudo que fala com o Arduino fica protegido: se a comunicacao cair
        #     (reset por brownout, cabo, etc.), reconecta e volta a procurar
        #     em vez de matar o programa em silencio (SystemExit do pymata4) ---
        try:
            if ultima_deteccao is not None and modo == "procurando":
                if processar_busca(pos):
                    # alinhado: mede a distancia e so coleta se o objeto estiver perto o bastante
                    distancia = ler_distancia()
                    if distancia is None:
                        print("Alinhado, mas sem leitura de distancia -> ignorando objeto")
                    elif distancia > DIST_MAX_COLETA:
                        print(f"Objeto a {distancia:.1f} cm (> {DIST_MAX_COLETA:.0f} cm) -> ignorando")
                    else:
                        distancia_coleta = distancia
                        modo = "pegando"

            if modo == "pegando":
                executar_coleta_e_descarte(distancia_coleta)
                ultima_deteccao = None
                resetar_busca()
                modo = "procurando"
                falhas_seguidas = 0  # ciclo completo com sucesso -> zera o contador

        except (Exception, SystemExit):
            falhas_seguidas += 1
            print(f"\n[ALERTA] Comunicacao com o Arduino caiu durante a operacao "
                  f"(falha {falhas_seguidas}/{MAX_FALHAS_SEGUIDAS}).")
            print("Causa mais comum: RESET do Arduino por queda de tensao (brownout) dos servos.")
            traceback.print_exc()

            if falhas_seguidas >= MAX_FALHAS_SEGUIDAS:
                print("\nFalhas demais seguidas -> parando. Verifique a ALIMENTACAO dos servos "
                      "(fonte externa 5-6V dedicada + GND comum com o Arduino).")
                break

            print("Tentando reconectar e voltar a procurar...")
            if not recuperar_board():
                print("Nao consegui reconectar -> parando.")
                break
            ultima_deteccao = None
            resetar_busca()
            modo = "procurando"
            print("Reconectado! Voltando a procurar...")

        cv2.putText(annotated, f"Modo: {modo}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated, f"Base: {angulo_atual['base']}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("ROBO", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nInterrompido pelo usuario (Ctrl+C).")

except Exception:
    # erro inesperado (nao relacionado a placa): mostra e segura a janela pra dar tempo de ler
    print("\n==================== ERRO INESPERADO ====================")
    traceback.print_exc()
    print("========================================================")
    input("Pressione ENTER para fechar...")

finally:
    # =========================
    # FINALIZACAO (sempre roda: erro, ESC ou fim normal)
    # =========================
    desligar_todos()  # corta o PWM pra nao deixar servo forcando/aquecendo
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    try:
        board.shutdown()
    except (Exception, SystemExit):
        pass
