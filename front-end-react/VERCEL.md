# Deploy na Vercel

Configure o projeto da Vercel apontando para esta pasta:

- Root Directory: `front-end-react`
- Framework Preset: `Vite`
- Install Command: `npm ci`
- Build Command: `npm run build`
- Output Directory: `dist`

Variável de ambiente obrigatória para produção:

```env
VITE_API_BASE_URL=https://api-f22.onrender.com
```

Sem essa variável, o frontend usa `https://api-f22.onrender.com` como fallback.

O arquivo `vercel.json` já inclui o rewrite necessário para rotas SPA como `/login`, `/cadastro`, `/sobre` e `/contato` funcionarem ao recarregar a página.
