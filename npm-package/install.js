#!/usr/bin/env node

const { execSync } = require('child_process');

const PYTHON_SOURCE = 'git+https://github.com/zapdev-labs/turboquant.git';
const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 8;

console.log('🔧 TurboQuant NPM Wrapper Setup\n');
function run(command, options = {}) {
  try {
    execSync(command, {
      stdio: options.silent ? 'ignore' : 'inherit',
    });
    return true;
  } catch {
    return false;
  }
}

function read(command) {
  try {
    return execSync(command, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return null;
  }
}

function parseVersion(versionOutput) {
  const match = versionOutput.match(/Python\s+(\d+)\.(\d+)(?:\.(\d+))?/i);
  if (!match) return null;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3] || 0),
  };
}

function isSupportedVersion(version) {
  if (version.major > MIN_PYTHON_MAJOR) return true;
  if (version.major < MIN_PYTHON_MAJOR) return false;
  return version.minor >= MIN_PYTHON_MINOR;
}

// Find Python executable
function findPython() {
  const pythonCommands = ['python3', 'python', 'py'];
  for (const cmd of pythonCommands) {
    const versionOutput = read(`${cmd} --version`);
    if (!versionOutput) {
      continue;
    }

    const version = parseVersion(versionOutput);
    if (!version) continue;
    if (isSupportedVersion(version)) {
      return {
        command: cmd,
        version: `${version.major}.${version.minor}.${version.patch}`,
      };
    }
  }

  for (const cmd of pythonCommands) {
    const versionOutput = read(`${cmd} --version`);
    if (!versionOutput) continue;
    const version = parseVersion(versionOutput);
    if (!version) continue;
    console.error(
      `Found ${cmd} (${version.major}.${version.minor}.${version.patch}) but TurboQuant requires Python 3.8+`
    );
    process.exit(1);
  }
  console.error('Python not found. Please install Python 3.8+');
  console.error('Visit: https://www.python.org/downloads/');
  process.exit(1);
}
function ensurePip(python) {
  if (run(`${python} -m pip --version`, { silent: true })) {
    return true;
  }

  console.log('⚠️  pip not found. Bootstrapping with ensurepip...\n');
  if (!run(`${python} -m ensurepip --upgrade`)) {
    return false;
  }

  return run(`${python} -m pip --version`, { silent: true });
}

function hasWorkingTurboQuantInstall(python) {
  return run(`${python} -c "import numpy; import turboquant"`, { silent: true });
}

function installTurboQuant(python) {
  const installCommands = [
    `${python} -m pip install --upgrade ${PYTHON_SOURCE}`,
    `${python} -m pip install --user --upgrade ${PYTHON_SOURCE}`,
    `${python} -m pip install --break-system-packages --upgrade ${PYTHON_SOURCE}`,
    `${python} -m pip install --user --break-system-packages --upgrade ${PYTHON_SOURCE}`,
  ];

  for (const command of installCommands) {
    console.log(`→ ${command}`);
    if (run(command)) {
      return true;
    }
  }

  return false;
}

function printManualFallback(python) {
  console.log('\nTo complete setup manually, run one of these commands:\n');
  console.log('Option 1 - Standard install:');
  console.log(`  ${python} -m pip install --upgrade ${PYTHON_SOURCE}\n`);
  console.log('Option 2 - User install (no sudo):');
  console.log(`  ${python} -m pip install --user --upgrade ${PYTHON_SOURCE}\n`);
  console.log('Option 3 - If pip is missing:');
  console.log('  Ubuntu/Debian: sudo apt install python3-pip');
  console.log('  Fedora/RHEL:   sudo dnf install python3-pip');
  console.log('  Arch:          sudo pacman -S python-pip');
  console.log(`  ${python} -m ensurepip --upgrade`);
  console.log(`  ${python} -m pip install --upgrade ${PYTHON_SOURCE}\n`);
}

// Main setup
async function main() {
  const pythonInfo = findPython();
  const python = pythonInfo.command;

  console.log(`✓ Found Python: ${python} (${pythonInfo.version})\n`);

  if (!ensurePip(python)) {
    console.log('⚠️  pip is unavailable and could not be initialized.');
    printManualFallback(python);
    console.log('✅ NPM wrapper is ready!');
    console.log('\nThe CLI requires the Python package to be installed first.');
    return;
  }

  if (hasWorkingTurboQuantInstall(python)) {
    console.log('✓ TurboQuant Python package (with numpy) is already installed\n');
  } else {
    console.log('⚠️  TurboQuant Python package (or dependencies like numpy) not found');
    console.log('Installing TurboQuant Python package and dependencies...\n');

    const installed = installTurboQuant(python);
    if (!installed || !hasWorkingTurboQuantInstall(python)) {
      console.log('\n⚠️  Automatic Python package installation failed.');
      printManualFallback(python);
    } else {
      console.log('\n✓ TurboQuant Python package installed and verified\n');
    }
  }
  
  console.log('✅ NPM wrapper is ready!');
  console.log('\nUsage:');
  console.log('  turboquant --help');
  console.log('  turboquant kv-analyze --model-size 70b');
  console.log('\nThe CLI requires the Python package to be installed and importable.');
}

main().catch(e => {
  console.error('Setup error:', e);
  process.exit(1);
});
