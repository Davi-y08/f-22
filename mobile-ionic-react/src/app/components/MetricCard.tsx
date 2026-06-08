import { IonCard, IonCardContent, IonIcon } from "@ionic/react";

type MetricCardProps = {
  icon: string;
  label: string;
  value: string | number;
};

function MetricCard({ icon, label, value }: MetricCardProps) {
  return (
    <IonCard className="metric-card">
      <IonCardContent>
        <IonIcon icon={icon} />
        <span>{label}</span>
        <strong>{value}</strong>
      </IonCardContent>
    </IonCard>
  );
}

export default MetricCard;
