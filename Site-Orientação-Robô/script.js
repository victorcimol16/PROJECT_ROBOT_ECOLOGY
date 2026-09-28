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
