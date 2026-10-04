import type { ReactNode } from "react";

/**
 * A gold outline round a panel's options with its Opacity on the top edge
 * (the style of MangoPrint Prepress's "Apply edits to" group): the one
 * control visibly covers everything inside. Double-click resets to 100%.
 */
export default function OpacityGroup({
  value,
  onChange,
  disabled,
  what,
  children,
}: {
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  /** Named in the tooltip: "Fade everything in <what> together". */
  what: string;
  children: ReactNode;
}) {
  const pct = value * 100;
  return (
    <div className="opacity-group">
      <label
        className="opacity-group__control"
        title={`Fade everything in ${what} together, like a layer's opacity (double-click: 100%)`}
      >
        <span className="opacity-group__label">Opacity</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={value}
          disabled={disabled}
          style={{ background: `linear-gradient(to right, #ffcc33 ${pct}%, #3a3b45 ${pct}%)` }}
          onChange={(e) => onChange(Number(e.target.value))}
          onDoubleClick={() => onChange(1)}
        />
        <span className="opacity-group__value">{Math.round(pct)}%</span>
      </label>
      {children}
    </div>
  );
}
