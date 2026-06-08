# Stealth Lens Mobile

Versao mobile em Ionic React do projeto Stealth Lens, consumindo a API hospedada em:

`https://api-f22.onrender.com`

## Funcionalidades

- Login e cadastro usando a API do projeto.
- Home com CRUD de cameras.
- Listagem com busca.
- Tela de detalhes da camera.
- Criacao de chave do distribuido.
- Paginas Sobre e Contato.
- Tratamento amigavel para erros de API e sessao expirada.
- Tema escuro seguindo a identidade visual do frontend React.

## Como rodar

```bash
npm install
npm run dev
```

## Como gerar build

```bash
npm run build
```

## Observacao

Se `VITE_API_BASE_URL` nao for informado ou estiver invalido, o app usa automaticamente a API padrao do projeto.
