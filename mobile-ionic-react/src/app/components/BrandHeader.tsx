import { IonAvatar, IonText } from "@ionic/react";
import logoUrl from "../../assets/stealth-lens-logo.jpg";

type BrandHeaderProps = {
  subtitle?: string;
};

function BrandHeader({ subtitle = "Scientia Vinces" }: BrandHeaderProps) {
  return (
    <div className="brand-header">
      <IonAvatar className="brand-avatar">
        <img src={logoUrl} alt="Stealth Lens" />
      </IonAvatar>
      <div>
        <IonText color="light">
          <h1>Stealth Lens</h1>
        </IonText>
        <IonText color="medium">
          <p>{subtitle}</p>
        </IonText>
      </div>
    </div>
  );
}

export default BrandHeader;
