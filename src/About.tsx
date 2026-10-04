import { useEffect } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Coffee, Globe, X } from "lucide-react";
import type { ModelInfo } from "./api";
import aboutPhoto from "./assets/about.jpg";
import { useAppVersion } from "./version";

export const COFFEE_URL = "https://buymeacoffee.com/chriscorkphotography";
const WEBSITE_URL = "https://www.chriscorkphotography.co.uk";
const INSTAGRAM_URL = "https://www.instagram.com/chriscorkphotography";
const FACEBOOK_URL = "https://www.facebook.com/chriscorkphoto";
const SOURCE_URL = "https://github.com/crispin81/RapidRetouch";
const LICENCE_URL = "https://www.gnu.org/licenses/agpl-3.0.html";
// Chris's photo for the left half (a portrait in a rapeseed field, 900x1200
// sRGB copy of DSCF7306-Edit). The watermark is laid over it here, not baked
// into the file.
const PHOTO_URL: string | null = aboutPhoto;

// What each AI model does in the app, by the job it's registered for.
const MODEL_ROLES: Record<string, string> = {
  subject_mask: "Finds the subject, for backdrop smoothing and the Neck, Body and Clothes tools",
  inpaint: "Fills in whatever the Remove brush takes away",
  face_landmarks: "Finds faces, eyes and lips for the Skin, Eyes, Mouth and Dodge & Burn tools",
  person_parts: "Tells skin from clothing, for the Neck, Body and Clothes tools",
};

// The main open-source projects the app is built on.
const BUILT_ON = [
  ["Tauri", "https://tauri.app"],
  ["React", "https://react.dev"],
  ["PyTorch", "https://pytorch.org"],
  ["OpenCV", "https://opencv.org"],
  ["MediaPipe", "https://ai.google.dev/edge/mediapipe"],
  ["LibRaw", "https://www.libraw.org"],
  ["NumPy", "https://numpy.org"],
] as const;

// The icon set has no brand logos: simple outline glyphs in its style.
function Instagram({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <rect x="3" y="3" width="18" height="18" rx="5" />
      <circle cx="12" cy="12" r="4" />
      <circle cx="17.5" cy="6.5" r="0.8" fill="currentColor" stroke="none" />
    </svg>
  );
}

function Facebook({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <path d="M15 3h-2.5A3.5 3.5 0 0 0 9 6.5V21M6 11h8" strokeLinecap="round" />
    </svg>
  );
}

function Link({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      onClick={(e) => {
        e.preventDefault();
        openUrl(href);
      }}
    >
      {children}
    </a>
  );
}

export default function About({ models, onClose }: { models: ModelInfo[]; onClose: () => void }) {
  const version = useAppVersion();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal" onClick={onClose}>
      <div className="modal__box about" onClick={(e) => e.stopPropagation()}>
        <button className="about__close" onClick={onClose} title="Close">
          <X size={15} />
        </button>
        <div className="about__photo">
          {PHOTO_URL ? (
            <img src={PHOTO_URL} alt="A portrait by Chris Cork Photography" draggable={false} />
          ) : (
            <div className="about__photo-empty">Photo</div>
          )}
          <div className="about__watermark">Chris Cork Photography</div>
        </div>
        <div className="about__text">
          <div className="about__head">
            <div>
              <h2>
                <span className="titlebar__title-rapid">Rapid</span>
                <span className="titlebar__title-retouch">Retouch</span>{" "}
                {version && <span className="titlebar__version">v{version}</span>}
              </h2>
              <p>
                Portrait retouching for photographers, made by Chris Cork, a portrait photographer. Everything
                runs on your own computer: your photos never leave it.
              </p>
              <div className="about__links">
                <Link href={WEBSITE_URL}>
                  <Globe size={14} /> Website
                </Link>
                <Link href={INSTAGRAM_URL}>
                  <Instagram size={14} /> Instagram
                </Link>
                <Link href={FACEBOOK_URL}>
                  <Facebook size={14} /> Facebook
                </Link>
              </div>
            </div>
          </div>

          <h3>Free and open source</h3>
          <p>
            RapidRetouch is free software under the <Link href={LICENCE_URL}>GNU Affero General Public License v3</Link>
            . You can use it, study it, change it and share it; if you share a changed version, share its source too.
            The <Link href={SOURCE_URL}>source code</Link> is on GitHub.
          </p>

          <h3>AI models</h3>
          <p className="about__small">
            All run on your computer. Each is downloaded from its makers the first time it's needed, and each allows
            commercial use, so it's safe for paid client work.
          </p>
          <ul className="about__models">
            {models.map((m) => (
              <li key={m.id}>
                <strong>{m.name}</strong> · <Link href={m.licence_url}>{m.licence}</Link>
                <br />
                <span className="about__small">{MODEL_ROLES[m.task] ?? m.task}</span>
              </li>
            ))}
          </ul>

          <h3>Built on</h3>
          <p className="about__small">
            {BUILT_ON.map(([name, url], i) => (
              <span key={name}>
                {i > 0 && " · "}
                <Link href={url}>{name}</Link>
              </span>
            ))}
            , and many other open-source projects. Thank you to everyone who makes them.
          </p>

          <div className="about__coffee">
            <p>
              RapidRetouch is free, and it'll stay free. It's also powered almost entirely by coffee: I drink a
              worrying amount of it while building this app. Every donation goes straight into the developer, and
              performance improves accordingly. ☕😄
            </p>
            <button onClick={() => openUrl(COFFEE_URL)}>
              <Coffee size={15} /> Buy me a coffee
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
