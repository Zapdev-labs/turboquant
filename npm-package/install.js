#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔧 TurboQuant NPM Wrapper Setup\n');

// Find Python executable
function findPython() {
  const pythonCommands = ['python3', 'python', 'py'];
  
  for (const cmd of pythonCommands) {
    try {
      require('child_process').execSync(`${cmd} --version`, { stdio: 'ignore' });
      return cmd;
    } catch (e) {
      continue;
    }
  }
  
  console.error('Python not found. Please install Python 3.8+');
  console.error('Visit: https://www.python.org/downloads/');
  process.exit(1);
}

// Main setup
async function main() {
  const python = findPython();
  
  console.log(`✓ Found Python: ${python}\n`);
  
  // Check if TurboQuant is already installed
  try {
    execSync(`${python} -c "import turboquant"`, { stdio: 'ignore' });
    console.log('✓ TurboQuant Python package is already installed\n');
  } catch (e) {
    console.log('⚠️  TurboQuant Python package not found');
    console.log('\nTo complete setup, run one of these commands:\n');
    console.log('Option 1 - Install from GitHub (recommended):');
    console.log(`  ${python} -m pip install git+https://github.com/zapdev-labs/turboquant.git\n`);
    console.log('Option 2 - Install from local source:');
    console.log(`  cd /path/to/turboquant && ${python} -m pip install -e .\n`);
    console.log('Option 3 - If pip is missing:');
    console.log(`  ${python} -m ensurepip --upgrade`);
    console.log(`  ${python} -m pip install git+https://github.com/zapdev-labs/turboquant.git\n`);
    console.log('After installing the Python package, the CLI will work.\n');
  }
  
  console.log('✅ NPM wrapper is ready!');
  console.log('\nUsage:');
  console.log('  turboquant --help');
  console.log('  turboquant kv-analyze --model-size 70b');
  console.log('\nThe CLI requires the Python package to be installed first.');
}

main().catch(e => {
  console.error('Setup error:', e);
  process.exit(1);
});
