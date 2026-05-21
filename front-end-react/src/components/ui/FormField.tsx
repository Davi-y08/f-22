import type {
  ChangeEventHandler,
  HTMLInputTypeAttribute,
  ReactNode,
  TextareaHTMLAttributes,
} from "react";
import type { LucideIcon } from "lucide-react";

interface FormFieldProps {
  autoComplete?: string;
  disabled?: boolean;
  icon?: LucideIcon;
  id: string;
  label: string;
  name: string;
  onChange: ChangeEventHandler<HTMLInputElement>;
  placeholder?: string;
  required?: boolean;
  trailingAction?: ReactNode;
  type?: HTMLInputTypeAttribute;
  value: string;
}

interface TextAreaFieldProps
  extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, "className"> {
  id: string;
  label: string;
}

function FormField({
  autoComplete,
  disabled = false,
  icon: Icon,
  id,
  label,
  name,
  onChange,
  placeholder = "",
  required = false,
  trailingAction,
  type = "text",
  value,
}: FormFieldProps) {
  return (
    <label className="grid gap-2 text-sm font-medium text-slate-200" htmlFor={id}>
      {label}
      <div className="relative">
        {Icon ? (
          <Icon className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-slate-500" />
        ) : null}
        <input
          autoComplete={autoComplete}
          className="input-shell min-h-11 w-full rounded-lg px-3 py-2.5 text-sm placeholder:text-slate-500 disabled:cursor-not-allowed disabled:opacity-55"
          disabled={disabled}
          id={id}
          name={name}
          onChange={onChange}
          placeholder={placeholder}
          required={required}
          style={{
            paddingLeft: Icon ? "2.45rem" : "0.75rem",
            paddingRight: trailingAction ? "3rem" : "0.75rem",
          }}
          type={type}
          value={value}
        />
        {trailingAction ? (
          <div className="absolute inset-y-0 right-2 flex items-center">
            {trailingAction}
          </div>
        ) : null}
      </div>
    </label>
  );
}

export function TextAreaField({ id, label, ...props }: TextAreaFieldProps) {
  return (
    <label className="grid gap-2 text-sm font-medium text-slate-200" htmlFor={id}>
      {label}
      <textarea
        className="input-shell min-h-32 w-full resize-y rounded-lg px-3 py-2.5 text-sm placeholder:text-slate-500"
        id={id}
        {...props}
      />
    </label>
  );
}

export default FormField;
