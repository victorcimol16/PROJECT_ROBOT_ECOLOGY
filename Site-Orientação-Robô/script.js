const botao = document.getElementById("botaoChamar");
const status = document.getElementById("status");
const robo = document.getElementById("robo");

// Pontos do trajeto (em %), seguindo a mesma trilha desenhada no mapa.
// Comeca no canto de baixo e termina ao lado do usuario.
const trajeto = [
  { left: "63%", top: "45%" }, // inicio (dentro do predio do CIMOL)
  { left: "63%", top: "50%" }, // desce um pouco
  { left: "41%", top: "50%" }  // anda para a esquerda ate o usuario
];

const posicaoInicial = trajeto[0];

botao.addEventListener("click", function () {
  botao.disabled = true;
  status.textContent = "Chamando o robo...";
  status.className = "status chamando";
  robo.classList.add("pulsando");

  // Anda ponto a ponto: cada passo espera a transicao (1.2s) terminar.
  let passo = 1;

  function andar() {
    if (passo < trajeto.length) {
      robo.style.left = trajeto[passo].left;
      robo.style.top = trajeto[passo].top;
      passo++;
      setTimeout(andar, 1200); // mesmo tempo da transicao no CSS
    } else {
      chegou();
    }
  }

  andar();
});

function chegou() {
  status.textContent = "O robo chegou!";
  status.className = "status chegou";
  robo.classList.remove("pulsando");

  // Depois de 3s, o robo volta para o inicio e libera o botao.
  setTimeout(function () {
    robo.style.left = posicaoInicial.left;
    robo.style.top = posicaoInicial.top;
    status.textContent = "Robo disponivel";
    status.className = "status";
    botao.disabled = false;
  }, 3000);
}

/* =====================================================================
   FORMULARIO DE PROBLEMAS E DUVIDAS + localStorage (Criterio 3)
   ===================================================================== */

const form = document.getElementById("formRelato");
const campoNome = document.getElementById("nome");
const campoTipo = document.getElementById("tipo");
const campoMensagem = document.getElementById("mensagem");
const aviso = document.getElementById("avisoForm");
const listaRelatos = document.getElementById("listaRelatos");
const botaoLimpar = document.getElementById("limparRelatos");

const CHAVE_STORAGE = "chamarRobo.relatos";
const CHAVE_RASCUNHO = "chamarRobo.rascunho";

// Le os relatos salvos no localStorage. Retorna sempre um array,
// mesmo que o dado esteja ausente ou corrompido (evita erro no Console).
function carregarRelatos() {
  try {
    const bruto = localStorage.getItem(CHAVE_STORAGE);
    const dados = bruto ? JSON.parse(bruto) : [];
    return Array.isArray(dados) ? dados : [];
  } catch (erro) {
    console.warn("Nao foi possivel ler os relatos salvos:", erro);
    return [];
  }
}

// Salva o array de relatos no localStorage.
function salvarRelatos(relatos) {
  try {
    localStorage.setItem(CHAVE_STORAGE, JSON.stringify(relatos));
  } catch (erro) {
    console.warn("Nao foi possivel salvar os relatos:", erro);
  }
}

// Salva o que esta digitado no formulario (rascunho), para nao sumir no F5.
function salvarRascunho() {
  try {
    const rascunho = {
      nome: campoNome.value,
      tipo: campoTipo.value,
      mensagem: campoMensagem.value
    };
    localStorage.setItem(CHAVE_RASCUNHO, JSON.stringify(rascunho));
  } catch (erro) {
    console.warn("Nao foi possivel salvar o rascunho:", erro);
  }
}

// Restaura o rascunho salvo nos campos do formulario (apos F5).
function restaurarRascunho() {
  try {
    const bruto = localStorage.getItem(CHAVE_RASCUNHO);
    if (!bruto) return;
    const rascunho = JSON.parse(bruto);
    if (!rascunho || typeof rascunho !== "object") return;
    campoNome.value = rascunho.nome || "";
    campoTipo.value = rascunho.tipo || "Problema";
    campoMensagem.value = rascunho.mensagem || "";
  } catch (erro) {
    console.warn("Nao foi possivel restaurar o rascunho:", erro);
  }
}

// Monta um texto de data legivel a partir de um timestamp.
function formatarData(timestamp) {
  const data = new Date(timestamp);
  return data.toLocaleString("pt-BR");
}

// Desenha a lista de relatos na tela a partir do localStorage.
function renderizarRelatos() {
  const relatos = carregarRelatos();
  listaRelatos.innerHTML = "";

  if (relatos.length === 0) {
    const vazio = document.createElement("li");
    vazio.className = "lista-vazia";
    vazio.textContent = "Nenhum relato enviado ainda.";
    listaRelatos.appendChild(vazio);
    return;
  }

  // Mostra os mais recentes primeiro.
  relatos
    .slice()
    .reverse()
    .forEach(function (relato) {
      const item = document.createElement("li");
      const classeTipo =
        relato.tipo === "Problema" ? "tipo-problema" : "tipo-duvida";
      item.className = "relato-item " + classeTipo;

      const cabecalho = document.createElement("div");
      cabecalho.className = "relato-cabecalho";

      const nome = document.createElement("span");
      nome.className = "relato-nome";
      nome.textContent = relato.nome;

      const tipo = document.createElement("span");
      tipo.className = "relato-tipo";
      tipo.textContent = relato.tipo;

      cabecalho.appendChild(nome);
      cabecalho.appendChild(tipo);

      const msg = document.createElement("p");
      msg.className = "relato-msg";
      msg.textContent = relato.mensagem;

      const dataEl = document.createElement("span");
      dataEl.className = "relato-data";
      dataEl.textContent = formatarData(relato.data);

      item.appendChild(cabecalho);
      item.appendChild(msg);
      item.appendChild(dataEl);
      listaRelatos.appendChild(item);
    });
}

// Envio do formulario.
form.addEventListener("submit", function (evento) {
  evento.preventDefault();

  const nome = campoNome.value.trim();
  const tipo = campoTipo.value;
  const mensagem = campoMensagem.value.trim();

  if (nome === "" || mensagem === "") {
    aviso.textContent = "Preencha o nome e a mensagem.";
    aviso.className = "aviso-form erro";
    return;
  }

  const relatos = carregarRelatos();
  relatos.push({
    nome: nome,
    tipo: tipo,
    mensagem: mensagem,
    data: Date.now()
  });
  salvarRelatos(relatos);

  form.reset();
  localStorage.removeItem(CHAVE_RASCUNHO); // limpa o rascunho apos enviar
  aviso.textContent = "Relato enviado com sucesso!";
  aviso.className = "aviso-form ok";
  renderizarRelatos();
});

// Botao para limpar todos os relatos.
botaoLimpar.addEventListener("click", function () {
  salvarRelatos([]);
  aviso.textContent = "";
  aviso.className = "aviso-form";
  renderizarRelatos();
});

// Salva o rascunho sempre que o usuario digita/muda algum campo.
campoNome.addEventListener("input", salvarRascunho);
campoTipo.addEventListener("change", salvarRascunho);
campoMensagem.addEventListener("input", salvarRascunho);

// Ao carregar a pagina: restaura o rascunho e mostra os relatos salvos (apos F5).
restaurarRascunho();
renderizarRelatos();
