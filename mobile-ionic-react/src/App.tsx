import {
  IonApp,
  IonIcon,
  IonLabel,
  IonRouterOutlet,
  IonTabBar,
  IonTabButton,
  IonTabs,
} from "@ionic/react";
import { IonReactRouter } from "@ionic/react-router";
import {
  cameraOutline,
  callOutline,
  homeOutline,
  informationCircleOutline,
  logInOutline,
} from "ionicons/icons";
import { Redirect, Route } from "react-router-dom";
import AboutPage from "./app/pages/AboutPage";
import AuthPage from "./app/pages/AuthPage";
import CameraDetailsPage from "./app/pages/CameraDetailsPage";
import ContactPage from "./app/pages/ContactPage";
import HomePage from "./app/pages/HomePage";

function App() {
  return (
    <IonApp>
      <IonReactRouter>
        <IonTabs>
          <IonRouterOutlet>
            <Route exact path="/home" component={HomePage} />
            <Route exact path="/camera/:id" component={CameraDetailsPage} />
            <Route exact path="/sobre" component={AboutPage} />
            <Route exact path="/contato" component={ContactPage} />
            <Route exact path="/login" component={AuthPage} />
            <Route
              exact
              path="/cadastro"
              render={() => <AuthPage initialMode="cadastro" />}
            />
            <Route exact path="/">
              <Redirect to="/home" />
            </Route>
          </IonRouterOutlet>

          <IonTabBar slot="bottom" className="app-tab-bar">
            <IonTabButton tab="home" href="/home">
              <IonIcon icon={homeOutline} />
              <IonLabel>Home</IonLabel>
            </IonTabButton>
            <IonTabButton tab="cameras" href="/home">
              <IonIcon icon={cameraOutline} />
              <IonLabel>Cameras</IonLabel>
            </IonTabButton>
            <IonTabButton tab="sobre" href="/sobre">
              <IonIcon icon={informationCircleOutline} />
              <IonLabel>Sobre</IonLabel>
            </IonTabButton>
            <IonTabButton tab="contato" href="/contato">
              <IonIcon icon={callOutline} />
              <IonLabel>Contato</IonLabel>
            </IonTabButton>
            <IonTabButton tab="login" href="/login">
              <IonIcon icon={logInOutline} />
              <IonLabel>Login</IonLabel>
            </IonTabButton>
          </IonTabBar>
        </IonTabs>
      </IonReactRouter>
    </IonApp>
  );
}

export default App;
