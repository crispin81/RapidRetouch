import appIcon from "./assets/app-icon.png";

/**
 * Shown while a released app sets up its AI engine on first launch (or after
 * an update): what it's doing, how far along, and nothing to click. If it
 * fails, what happened and what to do.
 */
export default function SetupScreen({
  message,
  fraction,
  error,
}: {
  message: string;
  fraction: number | null;
  error: string | null;
}) {
  return (
    <div className="setup">
      <div className="setup__box">
        <img className="setup__icon" src={appIcon} alt="" draggable={false} />
        <h1>
          <span className="titlebar__title-rapid">Rapid</span>
          <span className="titlebar__title-retouch">Retouch</span>
        </h1>
        {error ? (
          <>
            <p className="setup__error">{error}</p>
            <p className="setup__note">Close RapidRetouch and open it again to try again.</p>
          </>
        ) : (
          <>
            <p className="setup__message">{message}</p>
            <div className="setup__bar">
              <div
                className={fraction === null ? "setup__bar--busy" : undefined}
                style={fraction === null ? undefined : { width: `${Math.max(2, fraction * 100)}%` }}
              />
            </div>
            <p className="setup__note">
              The first time, RapidRetouch downloads its AI engine: about 3 GB on a PC with an NVIDIA graphics card
              (it uses the card for speed), about 1 GB otherwise. It only happens once; after that it starts straight
              away.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
