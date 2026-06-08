import {
  IonBackButton,
  IonButtons,
  IonHeader,
  IonTitle,
  IonToolbar,
} from "@ionic/react";

type PageToolbarProps = {
  backHref?: string;
  title: string;
};

function PageToolbar({ backHref, title }: PageToolbarProps) {
  return (
    <IonHeader translucent>
      <IonToolbar>
        {backHref ? (
          <IonButtons slot="start">
            <IonBackButton defaultHref={backHref} text="" />
          </IonButtons>
        ) : null}
        <IonTitle>{title}</IonTitle>
      </IonToolbar>
    </IonHeader>
  );
}

export default PageToolbar;
