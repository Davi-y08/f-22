# Stealth Lens Mobile

Aplicativo React Native/Expo do sistema F22. Ele usa a mesma API do site para login, listagem de câmeras, alertas e snapshots.

## Rodar localmente

```powershell
npm install
npm start
```

O app usa `https://api-f22.onrender.com` por padrão. Para trocar a API, ajuste `expo.extra.apiBaseUrl` em `app.json`.

## Push notifications

O app registra `ExpoPushToken` na API em `/notification-devices` depois do login. Para push remoto em aparelho físico, configure o `extra.eas.projectId` do Expo/EAS no `app.json`.

No Android, o Expo Go não suporta push remoto com `expo-notifications` a partir do SDK 53. Nesse ambiente o app desativa o módulo de notificações para continuar abrindo; para validar push remoto, use uma development build.

No backend, deixe estas variáveis definidas:

```env
PUBLIC_API_BASE_URL=https://api-f22.onrender.com
EXPO_PUSH_ENABLED=true
EXPO_PUSH_ENDPOINT=
EXPO_PUSH_ACCESS_TOKEN=
```

`PUBLIC_API_BASE_URL` é usada para montar a URL pública temporária do snapshot que aparece no `richContent.image` da notificação.
