interface Props {
  label: string;
  hint?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  /** Double-clicking the slider resets it to this. */
  defaultValue: number;
  disabled?: boolean;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

/** Labelled range slider with the gold fill used across the app. */
export default function Slider({
  label,
  hint,
  min,
  max,
  step,
  value,
  defaultValue,
  disabled,
  format = (v) => v.toFixed(2),
  onChange,
}: Props) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="slider" title={hint}>
      <span className="slider__head">
        <span>{label}</span>
        <span className="slider__value">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ background: `linear-gradient(to right, #ffcc33 ${pct}%, #3a3b45 ${pct}%)` }}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={() => onChange(defaultValue)}
      />
    </label>
  );
}
