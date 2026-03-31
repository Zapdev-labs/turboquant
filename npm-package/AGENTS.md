# NPM PACKAGE

CLI wrapper that installs TurboQuant Python library globally.

## OVERVIEW

Thin Node.js wrapper enabling `npm install -g turboquant` then `turboquant` command anywhere. Auto-detects Python, installs the Python package via pip, and delegates all commands.

## STRUCTURE

```
npm-package/
├── bin/
│   ├── turboquant        # Main CLI binary
│   └── tq                # Short alias
├── src/
│   └── tui.tsx           # Interactive Terminal UI (React)
├── install.js            # Post-install Python setup
├── test.js               # Basic validation tests
├── index.js              # Programmatic API
├── package.json          # NPM manifest
└── tsconfig.json         # TypeScript config
```

## WHERE TO LOOK

| Purpose | File |
|---------|------|
| CLI entry | `bin/turboquant` |
| Short alias | `bin/tq` |
| TUI source | `src/tui.tsx` |
| Python setup | `install.js` |
| Validation | `test.js` |
| Programmatic API | `index.js` |

## CONVENTIONS

### CLI Pattern
- Dual binaries: `turboquant` (full) and `tq` (short)
- Delegates 100% to Python CLI - no logic here
- Python detection order: `python3` → `python` → `py`

### TUI Mode
- Run without arguments to launch interactive TUI
- Requires Bun runtime (`~/.bun/bin/bun`)
- Falls back to CLI help if Bun unavailable

### Installation Flow
1. `npm install -g turboquant` runs `install.js`
2. Detects Python version (requires 3.8+)
3. Runs `pip install turboquant` if not present
4. All future `turboquant` calls delegate to Python

## ANTI-PATTERNS

- **NEVER** add business logic here - pure delegation wrapper
- **DON'T** duplicate Python CLI functionality
- **AVOID** additional npm dependencies (keep lightweight)

## COMMANDS

```bash
# Install globally
npm install -g turboquant

# Or use npx (no install)
npx turboquant kv-analyze --model-size 70b

# Run TUI (requires Bun)
tq

# Test
npm test                            # Runs test.js

# Build TypeScript
npm run build                       # Compiles src/tui.tsx to dist/
```

## DEPENDENCIES

**Runtime:**
- `@opentui/core` - Terminal UI framework
- `@opentui/react` - React bindings for TUI
- `react` - React 19

**Dev:**
- `@types/node` - Node.js type definitions
- `typescript` - TypeScript compiler

**System:**
- Node.js 14+ required
- Bun required for TUI mode

## NOTES

- Published to GitHub Packages (not npm registry)
- CI/CD triggered on release creation
- Wrapper script handles Python-not-found errors gracefully
- TUI built with React + @opentui/core (Bun-optimized)
