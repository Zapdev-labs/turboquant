/**
 * TurboQuant Node.js entry point.
 * Use the `tq` or `turboquant` CLI commands, or spawn the Python library directly.
 */

const { execSync, spawn } = require('child_process');

function findPython() {
  for (const cmd of ['python3', 'python', 'py']) {
    try { execSync(`${cmd} --version`, { stdio: 'ignore' }); return cmd; }
    catch {}
  }
  return null;
}

module.exports = {
  version: require('./package.json').version,

  isAvailable() {
    const python = findPython();
    if (!python) return false;
    try { execSync(`${python} -c "import turboquant"`, { stdio: 'ignore' }); return true; }
    catch { return false; }
  },

  /** Spawn a turboquant CLI command and return a child process. */
  run(args = [], options = {}) {
    const python = findPython();
    if (!python) throw new Error('Python not found');
    return spawn(python, ['-m', 'turboquant.cli', ...args], {
      stdio: 'inherit',
      ...options,
    });
  },
};
