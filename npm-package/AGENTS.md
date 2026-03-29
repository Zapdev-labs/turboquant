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
├── install.js            # Post-install Python setup
├── test.js               # Basic validation tests
└── package.json          # NPM manifest
```

## WHERE TO LOOK

| Purpose | File |
|---------|------|
| CLI entry | `bin/turboquant` |
| Short alias | `bin/tq` |
| Python setup | `install.js` |
| Validation | `test.js` |

## CONVENTIONS

### CLI Pattern
- Dual binaries: `turboquant` (full) and `tq` (short)
- Delegates 100% to Python CLI - no logic here
- Python detection order: `python3` → `python` → `py`

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

# Test
npm test                            # Runs test.js
```

## DEPENDENCIES

- `child_process` - For spawning Python process
- Node.js 14+ required

## NOTES

- Published to GitHub Packages (not npm registry)
- CI/CD triggered on release creation
- Wrapper script handles Python-not-found errors gracefully
