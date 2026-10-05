import gc
import os
import sys
import time
import traceback

import cv2
from ultralytics import YOLO
from pymata4.pymata4 import Pymata4

# =========================
# CONFIGURACOES
# =========================
MODEL_PATH = r"C:\Users\Acer\Desktop\desafio projeto\dataset\runs\detect\train-2\weights\best.pt"
PORTA = 'COM5'
CAMERA = 0
CONF = 0.6
FRAME_SKIP = 3  # roda a deteccao a cada N frames (reaproveita a ultima deteccao nos demais)

PINOS = {"base": 11, "ombro": 10, "ante": 9, "garra": 6}

# sensor ultrassonico HC-SR04 (mede a distancia ate o objeto pra calcular a descida)
PINO_TRIG = 12
PINO_ECHO = 13
DIST_MIN_CM, DIST_MAX_CM = 4, 40   # faixa util de leitura (fora disso = leitura descartada)
AMOSTRAS_DISTANCIA = 5             # numero de leituras pra tirar a mediana e filtrar ruido
DIST_MAX_COLETA = 14.0             # objeto acima disso e IGNORADO (nao entra em modo de coleta)
# o braco passa MAIS quanto mais LONGE o objeto (erro proporcional a distancia), entao a
# distancia-alvo da descida nao usa desconto fixo, e sim uma correcao linear:
#     alvo_cm = distancia_medida * FATOR_COLETA + AJUSTE_COLETA_CM
# FATOR_COLETA (<1): recua mais conforme a distancia cresce. MENOR = recua mais no longe.
# AJUSTE_COLETA_CM: ajuste fino constante (cm). MAIOR = mira mais longe (menos recuo em tudo).
FATOR_COLETA = 0.71
AJUSTE_COLETA_CM = 1.2

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
MAX_FALHAS_FRAME = 30            # leituras de frame falhando seguido antes de desistir da camera

# manutencao anti-acumulo (pensado pra rodar 24/7 num Raspberry Pi: CPU fraca, RAM apertada).
# Com o tempo o loop pode ficar lento por backlog da camera, cache do torch ou lixo nao coletado.
# A estrategia e: limpeza leve periodica + watchdog de RAM/latencia que, se detectar acumulo,
# limpa e reseta a camera; so reinicia o processo inteiro se a limpeza leve nao resolver.
INTERVALO_MANUTENCAO_S = 120     # limpeza leve periodica (gc + cache) a cada X s, mesmo sem sintoma
LIMITE_LATENCIA_S = 0.6          # loop demorando mais que isso por volta (media) = acumulo -> limpa+reseta
LIMITE_RAM_MB = 700              # RAM (RSS) acima disso dispara limpeza+reset. Ajuste ao modelo do Pi:
                                 #   Pi 3 (1GB): ~500 | Pi 4 2GB: ~700 | Pi 4 4GB+: ~1200
RAM_REINICIO_MB = 1000           # se, apos limpar, a RAM continuar acima disso -> candidato a reinicio
MANUTENCOES_SEM_RESOLVER = 3     # limpezas seguidas que nao resolveram -> reinicia o processo (ultimo recurso)


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
    alvo_cm = max(DIST_MIN_CM, distancia * FATOR_COLETA + AJUSTE_COLETA_CM)
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
# MANUTENCAO / ANTI-ACUMULO
# =========================
def abrir_camera():
    """
    Cria a captura da camera ja com a resolucao e com buffer de 1 frame.
    BUFFERSIZE=1 e o detalhe-chave do anti-travamento: sem isso, se o YOLO processar
    mais devagar que a camera entrega, os frames enfileiram no driver e a latencia
    cresce sem parar (parece que 'trava'). Com buffer 1, sempre pegamos o frame mais novo.
    O backend DSHOW so existe no Windows; no Raspberry (Linux) usa o padrao (V4L2).
    """
    if sys.platform.startswith("win"):
        c = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
    else:
        c = cv2.VideoCapture(CAMERA)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    try:
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # nem todo backend respeita, por isso o try
    except Exception:
        pass
    return c


def ler_ram_mb():
    """
    RSS do processo em MB. Usa psutil se existir; senao le /proc (Linux/Raspberry);
    retorna None se nao der pra medir (ai o watchdog usa so a latencia).
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            paginas_residentes = int(f.read().split()[1])
        return paginas_residentes * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return None


def limpar_memoria():
    """Limpeza leve: forca o coletor de lixo e libera o cache do torch (se houver CUDA)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():  # no Raspberry isto e False -> no-op, sem custo
            torch.cuda.empty_cache()
    except Exception:
        pass


def resetar_camera():
    """Fecha e reabre a camera (zera backlog/driver). Retorna True se voltou a dar frame."""
    global cap
    try:
        cap.release()
    except Exception:
        pass
    time.sleep(0.2)
    cap = abrir_camera()
    for _ in range(40):  # aquecimento rapido
        ok, _f = cap.read()
        if ok and _f is not None:
            return True
        time.sleep(0.03)
    return False


def reiniciar_processo():
    """
    Ultimo recurso: religa o programa do zero pra um estado 100% limpo.
    Fecha tudo com cuidado (servos, placa, camera) e substitui o processo atual.
    So deve ser chamado quando esta 'procurando' (nunca no meio de uma coleta).
    """
    print("[manutencao] acumulo persistente -> reiniciando o processo pra um estado limpo...")
    desligar_todos()
    try:
        board.shutdown()
    except (Exception, SystemExit):
        pass
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    os.execv(sys.executable, [sys.executable] + sys.argv)  # nao retorna


estado_manut = {
    "ultima_limpeza": 0.0,
    "ultima_checagem_ram": 0.0,
    "sem_resolver": 0,
    "lat_media": None,  # media movel da duracao do loop (exclui as iteracoes de coleta)
}


def registrar_latencia(dt):
    """Media movel exponencial da duracao do loop (suaviza picos de 1 frame)."""
    m = estado_manut
    m["lat_media"] = dt if m["lat_media"] is None else 0.9 * m["lat_media"] + 0.1 * dt


def manutencao(agora):
    """
    Chamada 1x por iteracao enquanto 'procurando'. Decide (de forma barata) se precisa
    limpar/resetar. Retorna "reiniciar" quando a limpeza leve nao resolve e o processo
    deve ser religado; caso contrario None.
    """
    m = estado_manut
    precisa_pesado = False
    motivo = None

    # 1) latencia: loop ficou lento (backlog da camera / CPU saturada)
    if m["lat_media"] is not None and m["lat_media"] > LIMITE_LATENCIA_S:
        precisa_pesado, motivo = True, f"latencia {m['lat_media']:.2f}s"

    # 2) RAM: checa no maximo 1x/s (ler /proc a cada frame e desperdicio)
    if agora - m["ultima_checagem_ram"] >= 1.0:
        m["ultima_checagem_ram"] = agora
        ram = ler_ram_mb()
        if ram is not None and ram > LIMITE_RAM_MB:
            precisa_pesado, motivo = True, f"RAM {ram:.0f}MB"

    # 3) sem sintoma: so a limpeza leve periodica, pra manter o gc em dia
    if not precisa_pesado:
        if agora - m["ultima_limpeza"] >= INTERVALO_MANUTENCAO_S:
            print("[manutencao] limpeza leve periodica")
            limpar_memoria()
            m["ultima_limpeza"] = agora
        return None

    # acumulo detectado: limpa + reseta a camera
    print(f"[manutencao] acumulo detectado ({motivo}) -> limpando e resetando a camera")
    limpar_memoria()
    camera_ok = resetar_camera()
    m["ultima_limpeza"] = agora
    m["lat_media"] = None  # zera a media depois do reset (o 1o loop pos-reset e lento de proposito)

    ram_depois = ler_ram_mb()
    nao_resolveu = (not camera_ok) or (ram_depois is not None and ram_depois > RAM_REINICIO_MB)
    if nao_resolveu:
        m["sem_resolver"] += 1
        detalhe = "camera nao voltou" if not camera_ok else f"RAM ainda {ram_depois:.0f}MB"
        print(f"[manutencao] nao resolveu ({detalhe}) "
              f"[{m['sem_resolver']}/{MANUTENCOES_SEM_RESOLVER}]")
        if m["sem_resolver"] >= MANUTENCOES_SEM_RESOLVER:
            return "reiniciar"
    else:
        m["sem_resolver"] = 0
    return None


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

cap = abrir_camera()
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

# aquecimento: com CAP_DSHOW no Windows os primeiros read() costumam falhar enquanto
# o driver inicializa. Tenta por ~3s antes de desistir, em vez de morrer no 1o frame.
print("Aquecendo a camera...")
aquecida = False
for _ in range(60):
    ok, _frame = cap.read()
    if ok and _frame is not None:
        aquecida = True
        break
    time.sleep(0.05)
if not aquecida:
    try:
        board.shutdown()
    except (Exception, SystemExit):
        pass
    try:
        cap.release()
    except Exception:
        pass
    input(
        f"\nA camera de indice {CAMERA} abriu, mas nao entregou nenhum frame.\n"
        "Causas comuns: outro programa usando a webcam (Teams/Zoom/navegador) ou indice errado.\n"
        "Feche quem estiver usando a camera ou tente outro valor para CAMERA (0, 1, 2...).\n"
        "Pressione ENTER para fechar..."
    )
    raise SystemExit(1)

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
falhas_frame = 0  # leituras de frame falhando seguidas (zera a cada frame bom)

try:
    while True:
        t_inicio = time.time()

        # manutencao anti-acumulo: so roda quando esta 'procurando', nunca no meio de uma
        # coleta (a coleta e uma unica iteracao longa e bloqueante).
        if modo == "procurando" and manutencao(t_inicio) == "reiniciar":
            reiniciar_processo()  # substitui o processo atual, nao retorna

        fez_coleta = False  # marca iteracoes de coleta pra NAO poluir a media de latencia

        ok, frame = cap.read()
        if not ok or frame is None:
            # uma falha isolada e normal (driver engasga); so desiste se persistir
            falhas_frame += 1
            if falhas_frame >= MAX_FALHAS_FRAME:
                print("Falha ao ler frame da camera (persistente) -> encerrando.")
                break
            time.sleep(0.05)
            continue
        falhas_frame = 0

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
                fez_coleta = True
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

        if not fez_coleta:
            registrar_latencia(time.time() - t_inicio)

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
