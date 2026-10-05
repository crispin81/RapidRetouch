import { Eye, EyeOff } from "lucide-react";

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
  /** A colour scale (CSS gradient) the track itself shows, for sliders that
   * move along colours (e.g. hue, temperature); no gold fill then. */
  scale?: string;
  onChange: (v: number) => void;
  /** Switched off by its eye: the photo is shown without it (value kept). */
  off?: boolean;
  /** Given: the slider has an eye that switches it off and on. */
  onToggleOff?: () => void;
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
  scale,
  onChange,
  off,
  onToggleOff,
}: Props) {
  const pct = ((value - min) / (max - min)) * 100;
  const fill = centred
    ? `linear-gradient(to right, #3a3b45 ${Math.min(pct, 50)}%, #ffcc33 ${Math.min(pct, 50)}%, #ffcc33 ${Math.max(pct, 50)}%, #3a3b45 ${Math.max(pct, 50)}%)`
    : `linear-gradient(to right, #ffcc33 ${pct}%, #3a3b45 ${pct}%)`;
  return (
    <label className={`slider${scale ? " slider--scale" : ""}${off ? " slider--off" : ""}`} title={hint}>
      <span className="slider__head">
        <span className="slider__name">
          {onToggleOff && (
            <button
              type="button"
              className="slider__eye"
              disabled={disabled}
              onClick={(e) => {
                e.preventDefault(); // inside the label: don't also focus the slider
                onToggleOff();
              }}
              title={off ? `Switch ${label} back on` : `Switch ${label} off to compare (its value is kept)`}
            >
              {off ? <EyeOff size={13} /> : <Eye size={13} />}
            </button>
          )}
          {label}
        </span>
        <span className="slider__value">{off ? "Off" : format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ background: scale ?? fill }}
        disabled={disabled}
        onChange={(e) => {
          if (off && onToggleOff) onToggleOff(); // moving it switches it back on
          onChange(Number(e.target.value));
        }}
        onDoubleClick={() => onChange(defaultValue)}
      />
    </label>
  );
}
