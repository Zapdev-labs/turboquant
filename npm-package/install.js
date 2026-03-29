#!/usr/bin/env node

/**
 * Installation script for TurboQuant npm package
 * Checks Python availability and installs pip package if needed
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔧 Setting up TurboQuant...\n');

// Check Python version
function checkPython() {
  try {
    const version = execSync('python3 --version', { encoding: 'utf8' }).trim();
    console.log(`✓ Found: ${version}`);
    
    // Parse version number
    const match = version.match(/Python (\d+)\.(\d+)/);
    if (match) {
      const major = parseInt(match[1]);
      const minor = parseInt(match[2]);
      
      if (major < 3 || (major === 3 && minor < 8)) {
        console.error('✗ Python 3.8+ is required');
        process.exit(1);
      }
    }
    
    return 'python3';
  } catch (e) {
    try {
      const version = execSync('python --version', { encoding: 'utf8' }).trim();
      console.log(`✓ Found: ${version}`);
      return 'python';
    } catch (e2) {
      console.error('✗ Python not found. Please install Python 3.8+');
      console.error('  Visit: https://www.python.org/downloads/');
      process.exit(1);
    }
  }
}

// Check pip
function checkPip(python) {
  try {
    execSync(`${python} -m pip --version`, { stdio: 'ignore' });
    console.log('✓ pip is available');
    return true;
  } catch (e) {
    console.error('✗ pip not found');
    return false;
  }
}

// Install turboquant via pip
function installTurboQuant(python) {
  console.log('\n📦 Installing TurboQuant Python package...');
  
  try {
    // Try to install from PyPI (if published)
    execSync(`${python} -m pip install turboquant --quiet`, {
      stdio: 'inherit'
    });
    console.log('✓ TurboQuant installed from PyPI');
  } catch (e) {
    // Fallback: install from local directory
    console.log('Installing from local source...');
    const parentDir = path.resolve(__dirname, '..');
    
    try {
      execSync(`${python} -m pip install -e ${parentDir} --quiet`, {
        stdio: 'inherit'
      });
      console.log('✓ TurboQuant installed from source');
    } catch (e2) {
      console.error('✗ Failed to install TurboQuant');
      console.error('  You can install manually:');
      console.error(`  ${python} -m pip install -e .`);
      process.exit(1);
    }
  }
}

// Main installation flow
async function main() {
  const python = checkPython();
  
  if (!checkPip(python)) {
    console.error('\nPlease install pip first:');
    console.error(`  ${python} -m ensurepip --upgrade`);
    process.exit(1);
  }
  
  // Check if already installed
  try {
    execSync(`${python} -c "import turboquant"`, { stdio: 'ignore' });
    console.log('✓ TurboQuant is already installed\n');
  } catch (e) {
    installTurboQuant(python);
  }
  
  // Test installation
  console.log('\n🧪 Testing installation...');
  try {
    execSync(`${python} -m turboquant.cli --help`, { stdio: 'ignore' });
    console.log('✓ TurboQuant CLI is working!\n');
  } catch (e) {
    console.error('✗ CLI test failed');
    process.exit(1);
  }
  
  console.log('🎉 TurboQuant is ready to use!');
  console.log('\nQuick start:');
  console.log('  turboquant --help');
  console.log('  turboquant quick input.npy');
  console.log('  turboquant kv-analyze --model-size 70b\n');
}

main().catch(e => {
  console.error('Installation error:', e);
  process.exit(1);
});
