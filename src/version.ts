import { useEffect, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";

/** The app's version (from tauri.conf.json), e.g. "1.0.0"; "" until known. */
export function useAppVersion(): string {
  const [version, setVersion] = useState("");
  useEffect(() => {
    getVersion()
      .then(setVersion)
      .catch(() => undefined);
  }, []);
  return version;
}
