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
  /** Two-sided slider (e.g. matte .. gloss): the fill grows from the centre. */
  centred?: boolean;
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
  centred,
  onChange,
}: Props) {
  const pct = ((value - min) / (max - min)) * 100;
  const fill = centred
    ? `linear-gradient(to right, #3a3b45 ${Math.min(pct, 50)}%, #ffcc33 ${Math.min(pct, 50)}%, #ffcc33 ${Math.max(pct, 50)}%, #3a3b45 ${Math.max(pct, 50)}%)`
    : `linear-gradient(to right, #ffcc33 ${pct}%, #3a3b45 ${pct}%)`;
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
        style={{ background: fill }}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        onDoubleClick={() => onChange(defaultValue)}
      />
    </label>
  );
}
