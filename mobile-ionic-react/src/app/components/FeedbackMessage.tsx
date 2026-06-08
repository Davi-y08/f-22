import { IonIcon } from "@ionic/react";
import { alertCircleOutline, checkmarkCircleOutline } from "ionicons/icons";

type FeedbackMessageProps = {
  message: string;
  tone?: "danger" | "success";
};

function FeedbackMessage({ message, tone = "danger" }: FeedbackMessageProps) {
  if (!message) {
    return null;
  }

  return (
    <div className={`feedback feedback-${tone}`} role="status">
      <IonIcon icon={tone === "danger" ? alertCircleOutline : checkmarkCircleOutline} />
      <span>{message}</span>
    </div>
  );
}

export default FeedbackMessage;
