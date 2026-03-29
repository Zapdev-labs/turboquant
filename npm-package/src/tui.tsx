import { createCliRenderer } from "@opentui/core";
import { createRoot, useKeyboard } from "@opentui/react";
import { useState, useEffect, useRef } from "react";
import { spawn } from "child_process";

// ─── Types ────────────────────────────────────────────────────────────────────

interface HFModel {
  id: string;
  downloads: number;
  likes: number;
  tags: string[];
  private: boolean;
}

type Screen = "home" | "download" | "downloading";
type DownloadField = "search" | "results" | "token" | "bits" | "format" | "go";

const BITS_OPTIONS  = ["2-bit", "3-bit", "4-bit"] as const;
const BITS_VALUES   = ["2", "3", "4"] as const;
const FORMAT_OPTIONS = ["gguf", "safetensors"] as const;

// ─── Utils ────────────────────────────────────────────────────────────────────

function findPython(): string | null {
  const { execSync } = require("child_process") as typeof import("child_process");
  for (const cmd of ["python3", "python", "py"]) {
    try { execSync(`${cmd} --version`, { stdio: "ignore" }); return cmd; }
    catch {}
  }
  return null;
}

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

async function searchHF(query: string, token?: string): Promise<HFModel[]> {
  const params = new URLSearchParams({
    search: query,
    limit: "18",
    sort: "downloads",
    direction: "-1",
  });
  const headers: Record<string, string> = { Accept: "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`https://huggingface.co/api/models?${params}`, { headers });
  if (!res.ok) throw new Error(`HF API ${res.status}`);
  return res.json() as Promise<HFModel[]>;
}

// ─── Logo ─────────────────────────────────────────────────────────────────────

function Logo() {
  return (
    <box style={{ flexDirection: "column" }}>
      <text fg="#7C3AED" attributes={1}>  ████████╗██╗   ██╗██████╗ ██████╗  ██████╗ </text>
      <text fg="#7C3AED" attributes={1}>  ╚══██╔══╝██║   ██║██╔══██╗██╔══██╗██╔═══██╗</text>
      <text fg="#9F67FF" attributes={1}>     ██║   ██║   ██║██████╔╝██████╔╝██║   ██║</text>
      <text fg="#9F67FF" attributes={1}>     ██║   ██║   ██║██╔══██╗██╔══██╗██║▄▄ ██║</text>
      <text fg="#C084FC" attributes={1}>     ██║   ╚██████╔╝██║  ██║██████╔╝╚██████╔╝</text>
      <text fg="#C084FC" attributes={1}>     ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚══▀▀═╝ </text>
      <text fg="#4B5563">  ─────────── Extreme AI Model Compression ───────────</text>
    </box>
  );
}

// ─── Home Screen ──────────────────────────────────────────────────────────────

function HomeScreen({ onNavigate }: { onNavigate: (s: Screen) => void }) {
  const [sel, setSel] = useState(0);
  const menu = [
    { icon: "⬇", label: "Download Model",        action: () => onNavigate("download") },
    { icon: "⚡", label: "Quick Compress (.npy)",  action: () => {} },
    { icon: "📊", label: "KV Cache Analysis",     action: () => {} },
    { icon: "✕",  label: "Exit",                  action: () => process.exit(0) },
  ];

  useKeyboard((key) => {
    if (key.name === "up"    || key.name === "k") setSel((s) => Math.max(0, s - 1));
    if (key.name === "down"  || key.name === "j") setSel((s) => Math.min(menu.length - 1, s + 1));
    if (key.name === "return"|| key.name === "enter") menu[sel].action();
    if (key.name === "q"     || (key.ctrl && key.name === "c")) process.exit(0);
  });

  return (
    <box style={{ flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1 }}>
      <Logo />
      <box style={{ height: 1 }} />
      <box border borderStyle="rounded" title=" Main Menu " style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1 }}>
        {menu.map((item, i) => (
          <box key={i} style={{ paddingLeft: 1, paddingRight: 1 }}>
            {i === sel
              ? <text fg="#7C3AED" attributes={1}>{`▶ ${item.icon}  ${item.label}`}</text>
              : <text fg="#6B7280">{`  ${item.icon}  ${item.label}`}</text>}
          </box>
        ))}
      </box>
      <box style={{ paddingTop: 1 }}>
        <text fg="#374151">  ↑↓  navigate    enter  select    q  quit</text>
      </box>
    </box>
  );
}

// ─── Download Screen ──────────────────────────────────────────────────────────

function DownloadScreen({
  onBack,
  onStart,
}: {
  onBack: () => void;
  onStart: (args: { model: string; bits: string; format: string; token: string }) => void;
}) {
  const [field, setField]         = useState<DownloadField>("search");
  const [query, setQuery]         = useState("");
  const [results, setResults]     = useState<HFModel[]>([]);
  const [resultIdx, setResultIdx] = useState(0);
  const [status, setStatus]       = useState<"idle" | "loading" | "error">("idle");
  const [selected, setSelected]   = useState<HFModel | null>(null);
  const [tokenVal, setTokenVal]   = useState("");
  const [bitsIdx, setBitsIdx]     = useState(1);
  const [fmtIdx, setFmtIdx]       = useState(0);
  const debounceRef               = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced search whenever query changes
  useEffect(() => {
    if (!query.trim()) { setResults([]); setStatus("idle"); return; }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setStatus("loading");
    debounceRef.current = setTimeout(async () => {
      try {
        const hits = await searchHF(query, tokenVal || undefined);
        setResults(hits);
        setResultIdx(0);
        setStatus("idle");
      } catch {
        setStatus("error");
        setResults([]);
      }
    }, 450);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [query]);

  const TAB_ORDER: DownloadField[] = ["search", "results", "token", "bits", "format", "go"];

  useKeyboard((key) => {
    if (key.ctrl && key.name === "c") process.exit(0);
    if (key.name === "escape") {
      if (field === "results") { setField("search"); return; }
      onBack();
      return;
    }

    // Search box — always typing mode
    if (field === "search") {
      if (key.name === "backspace") { setQuery((s) => s.slice(0, -1)); return; }
      if (key.name === "tab") { setField(results.length ? "results" : "token"); return; }
      if ((key.name === "down" || key.name === "j") && results.length) {
        setField("results");
        return;
      }
      if (key.name === "return" || key.name === "enter") {
        // Treat as direct model ID if it looks like owner/model
        if (query.includes("/")) {
          setSelected({ id: query, downloads: 0, likes: 0, tags: [], private: false });
          setField("token");
        } else if (results.length) {
          setField("results");
        }
        return;
      }
      if (key.name && key.name.length === 1) { setQuery((s) => s + key.name); return; }
      return;
    }

    // Results list
    if (field === "results") {
      if (key.name === "up" || key.name === "k") {
        if (resultIdx === 0) { setField("search"); return; }
        setResultIdx((s) => Math.max(0, s - 1));
        return;
      }
      if (key.name === "down" || key.name === "j") {
        setResultIdx((s) => Math.min(results.length - 1, s + 1));
        return;
      }
      if (key.name === "return" || key.name === "enter") {
        setSelected(results[resultIdx]);
        setField("token");
        return;
      }
      if (key.name === "tab") { setField("token"); return; }
      return;
    }

    // Token field — typing mode
    if (field === "token") {
      if (key.name === "backspace") { setTokenVal((s) => s.slice(0, -1)); return; }
      if (key.name === "tab") { setField("bits"); return; }
      if (key.name === "return" || key.name === "enter") { setField("bits"); return; }
      if (key.name && key.name.length === 1) { setTokenVal((s) => s + key.name); return; }
      return;
    }

    if (key.name === "tab") {
      const i = TAB_ORDER.indexOf(field);
      setField(TAB_ORDER[(i + 1) % TAB_ORDER.length]);
      return;
    }

    if (field === "bits") {
      if (key.name === "left"  || key.name === "h") setBitsIdx((s) => Math.max(0, s - 1));
      if (key.name === "right" || key.name === "l") setBitsIdx((s) => Math.min(BITS_OPTIONS.length - 1, s + 1));
    }
    if (field === "format") {
      if (key.name === "left"  || key.name === "h") setFmtIdx((s) => Math.max(0, s - 1));
      if (key.name === "right" || key.name === "l") setFmtIdx((s) => Math.min(FORMAT_OPTIONS.length - 1, s + 1));
    }
    if (field === "go" && (key.name === "return" || key.name === "enter")) {
      const model = selected?.id ?? (query.includes("/") ? query : "");
      if (model) onStart({ model, bits: BITS_VALUES[bitsIdx], format: FORMAT_OPTIONS[fmtIdx], token: tokenVal });
    }
  });

  const on  = (f: DownloadField) => field === f;
  const bc  = (f: DownloadField): object => on(f) ? { borderColor: "#7C3AED" } : {};
  const model = selected?.id ?? (query.includes("/") ? query : "");

  // Status line for search box
  let searchHint = "";
  if (status === "loading") searchHint = " searching…";
  else if (status === "error") searchHint = " ✗ network error";
  else if (results.length) searchHint = ` ${results.length} results`;

  return (
    <box style={{ flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1 }}>
      <text fg="#C084FC" attributes={1}>  ⬇  Download &amp; Quantize Model</text>
      <text fg="#4B5563">  Type to search HuggingFace  ·  ESC back  ·  TAB next field</text>
      <box style={{ height: 1 }} />

      {/* Search box */}
      <box border borderStyle="rounded" title=" Search HuggingFace " style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, ...bc("search") }}>
        <text fg="#6B7280">🔍 </text>
        <text fg={on("search") ? "#E2E8F0" : "#9CA3AF"}>
          {query || (on("search") ? "" : "model name or owner/model-id")}
          {on("search") ? "█" : ""}
        </text>
        {searchHint ? <text fg="#4B5563">{searchHint}</text> : null}
      </box>

      {/* Results list */}
      {results.length > 0 && (
        <box border borderStyle="rounded" title=" Results " style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1, ...bc("results") }}>
          {results.slice(0, 8).map((m, i) => {
            const isSel = on("results") && i === resultIdx;
            const isChosen = selected?.id === m.id;
            const tags = m.tags.filter((t) => ["gguf", "safetensors", "gptq", "awq"].includes(t)).join(" ");
            return (
              <box key={m.id} style={{ flexDirection: "row" }}>
                <text fg={isSel ? "#7C3AED" : isChosen ? "#22C55E" : "#6B7280"} attributes={isSel || isChosen ? 1 : 0}>
                  {isSel ? "  ▶ " : isChosen ? "  ✓ " : "    "}
                </text>
                <text fg={isSel ? "#E2E8F0" : isChosen ? "#22C55E" : "#9CA3AF"} attributes={isSel ? 1 : 0}>
                  {m.id}
                </text>
                {tags ? <text fg="#4B5563">{`  [${tags}]`}</text> : null}
                <text fg="#374151">{`  ↓${fmtNum(m.downloads)}`}</text>
              </box>
            );
          })}
          {results.length > 8 && (
            <text fg="#374151">{`    … ${results.length - 8} more — refine your search`}</text>
          )}
        </box>
      )}

      {/* Selected model indicator */}
      {selected && (
        <box border borderStyle="rounded" title=" Selected " style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1 }}>
          <text fg="#22C55E" attributes={1}>✓ </text>
          <text fg="#E2E8F0">{selected.id}</text>
          {selected.downloads > 0 && <text fg="#4B5563">{`  ↓${fmtNum(selected.downloads)}`}</text>}
        </box>
      )}

      {/* HF Token */}
      <box border borderStyle="rounded" title=" HuggingFace Token (optional — required for gated models) " style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, ...bc("token") }}>
        <text fg={on("token") ? "#E2E8F0" : "#9CA3AF"}>
          {tokenVal
            ? `hf_${"·".repeat(Math.min(tokenVal.replace(/^hf_/, "").length, 30))}`
            : (on("token") ? "" : "hf_… (press TAB to skip)")}
          {on("token") ? "█" : ""}
        </text>
      </box>

      <box style={{ flexDirection: "row", gap: 2 }}>
        {/* Bits */}
        <box border borderStyle="rounded" title=" Quantization " style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, flexGrow: 1, ...bc("bits") }}>
          <text fg="#4B5563">◀ </text>
          {BITS_OPTIONS.map((b, i) => (
            <text key={i} fg={i === bitsIdx ? "#7C3AED" : "#4B5563"} attributes={i === bitsIdx ? 1 : 0}>
              {b + (i < BITS_OPTIONS.length - 1 ? "  " : "")}
            </text>
          ))}
          <text fg="#4B5563"> ▶</text>
        </box>

        {/* Format */}
        <box border borderStyle="rounded" title=" Format " style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1, flexGrow: 1, ...bc("format") }}>
          <text fg="#4B5563">◀ </text>
          {FORMAT_OPTIONS.map((f, i) => (
            <text key={i} fg={i === fmtIdx ? "#7C3AED" : "#4B5563"} attributes={i === fmtIdx ? 1 : 0}>
              {f + (i < FORMAT_OPTIONS.length - 1 ? "  " : "")}
            </text>
          ))}
          <text fg="#4B5563"> ▶</text>
        </box>
      </box>

      {/* Go button */}
      <box border borderStyle="rounded" style={{ paddingLeft: 2, paddingRight: 2, alignSelf: "flex-start", ...(on("go") ? { borderColor: "#7C3AED", backgroundColor: "#1E1B4B" } : {}) }}>
        <text fg={on("go") ? "#C084FC" : (model ? "#6B7280" : "#374151")} attributes={on("go") ? 1 : 0}>
          {on("go") ? "▶  DOWNLOAD & QUANTIZE  ◀" : (model ? "   download & quantize   " : "   select a model first  ")}
        </text>
      </box>

      <box style={{ paddingTop: 1 }}>
        <text fg="#374151">  type search · ↓ into results · enter select · TAB next</text>
      </box>
    </box>
  );
}

// ─── Downloading Screen ───────────────────────────────────────────────────────

function DownloadingScreen({ model, bits, format, token }: {
  model: string; bits: string; format: string; token: string;
}) {
  const [lines, setLines]       = useState<string[]>([`📥  ${model}`, `    ${bits}-bit · ${format}`, ""]);
  const [progress, setProgress] = useState(0);
  const [done, setDone]         = useState(false);
  const [failed, setFailed]     = useState(false);

  useEffect(() => {
    const python = findPython();
    if (!python) {
      setLines((l) => [...l, "✗ Python not found. Install Python 3.8+"]);
      setFailed(true);
      return;
    }

    const args = ["-m", "turboquant.cli", "download", model, "--bits", bits, "--format", format];
    if (token) args.push("--hf-token", token);

    const child = spawn(python, args, { env: process.env });
    const push  = (line: string) => setLines((l) => [...l.slice(-40), line]);

    child.stdout?.on("data", (d: Buffer) => {
      d.toString().split("\n").filter(Boolean).forEach((line) => {
        push(line);
        if (line.includes("okenizer"))              setProgress(10);
        if (line.includes("Loading model"))         setProgress(25);
        if (line.includes("Model loaded"))          setProgress(55);
        if (line.includes("Exporting"))             setProgress(70);
        if (line.includes("✓ Export") || line.includes("Saving")) setProgress(88);
        if (line.includes("✓ Model saved") || line.includes("Complete")) setProgress(100);
      });
    });

    child.stderr?.on("data", (d: Buffer) => {
      d.toString().split("\n").filter(Boolean).forEach((line) => {
        if (!line.includes("UserWarning") && !line.includes("FutureWarning") && !line.includes("tqdm")) {
          push("  " + line.trim());
        }
      });
    });

    child.on("close", (code: number | null) => {
      if (code === 0) {
        setProgress(100); setDone(true);
        push(""); push("✓ Complete!  Saved to ./models/");
      } else {
        setFailed(true);
        push(""); push(`✗ Exited with code ${code}`);
      }
    });

    return () => { child.kill(); };
  }, []);

  useKeyboard((key) => {
    if ((done || failed) && (key.name === "q" || key.name === "escape")) process.exit(0);
    if (key.ctrl && key.name === "c") process.exit(0);
  });

  const barWidth = 44;
  const filled   = Math.round((progress / 100) * barWidth);
  const empty    = barWidth - filled;

  return (
    <box style={{ flexDirection: "column", paddingLeft: 2, paddingRight: 2, paddingTop: 1, paddingBottom: 1 }}>
      <text fg="#C084FC" attributes={1}>  ⬇  Downloading &amp; Quantizing</text>
      <box style={{ height: 1 }} />
      <box border borderStyle="rounded" title=" Progress " style={{ flexDirection: "row", paddingLeft: 1, paddingRight: 1 }}>
        <text fg="#7C3AED">{"█".repeat(filled)}</text>
        <text fg="#3B3B5C">{"░".repeat(empty)}</text>
        <text fg="#9CA3AF">{`  ${progress}%`}</text>
      </box>
      <box border borderStyle="rounded" title=" Output " style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1, paddingTop: 1, height: 18 }}>
        {lines.slice(-15).map((line, i) => (
          <text key={i} fg={line.startsWith("✓") ? "#22C55E" : line.startsWith("✗") ? "#EF4444" : "#9CA3AF"}>
            {line}
          </text>
        ))}
      </box>
      {(done || failed) && (
        <box style={{ paddingTop: 1 }}>
          <text fg="#4B5563">  Press q or ESC to exit</text>
        </box>
      )}
    </box>
  );
}

// ─── App Root ─────────────────────────────────────────────────────────────────

function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [dlArgs, setDlArgs] = useState<{ model: string; bits: string; format: string; token: string } | null>(null);

  if (screen === "home")
    return <HomeScreen onNavigate={setScreen} />;
  if (screen === "download")
    return <DownloadScreen onBack={() => setScreen("home")} onStart={(args) => { setDlArgs(args); setScreen("downloading"); }} />;
  if (screen === "downloading" && dlArgs)
    return <DownloadingScreen {...dlArgs} />;
  return <HomeScreen onNavigate={setScreen} />;
}

// ─── Entry ────────────────────────────────────────────────────────────────────

const renderer = await createCliRenderer({ targetFps: 30 });
createRoot(renderer).render(<App />);
